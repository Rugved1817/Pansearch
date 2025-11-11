import os
import logging
from typing import List, Optional
import re
import json
import urllib.parse
import urllib.request
from io import BytesIO
from datetime import datetime
from openai import OpenAI
import httpx
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import regex

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

try:
    from .config import THRESHOLDS as TH, SEARCH as SP  # type: ignore
    from .utils import nlp  # type: ignore
    from .search import (
        search_by_pan as _search_by_pan,
        search_by_seed_name as _search_by_seed_name,
        fetch_rows_for_pan as _fetch_rows_for_pan,
        pick_canonical_names as _pick_canonical_names,
    )  # type: ignore
except ImportError:
    from config import THRESHOLDS as TH, SEARCH as SP
    from utils import nlp
    from search import (
        search_by_pan as _search_by_pan,
        search_by_seed_name as _search_by_seed_name,
        fetch_rows_for_pan as _fetch_rows_for_pan,
        pick_canonical_names as _pick_canonical_names,
    )

DB_PATH = os.getenv("TX_DB_PATH", "data/tx.duckdb")

# Basic logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pansearch.api")

# Global font registration cache
_DEVANAGARI_FONTS = None

app = FastAPI(title="Transactions Search API", version="0.1.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


def is_devanagari(text: str) -> bool:
	"""Return True if any character is Devanagari."""
	return bool(re.search(r"[\u0900-\u097F]", text or ""))


def clean_english_name(raw_english: str) -> str:
	"""
	Clean ITRANS generated English text to title-case names without trailing schwa.
	"""
	if not raw_english:
		return ""
	raw_english = raw_english.replace("\u00A0", " ").replace("\xa0", " ").strip()
	cleaned_words: list[str] = []
	for word in raw_english.split():
		w = re.sub(r"(?<=[^aeiou])a$", "", word, flags=re.IGNORECASE)
		w = w.lower().capitalize()
		cleaned_words.append(w)
	return " ".join(cleaned_words)


def generate_english_spelling_variations(name: str, rules: dict[str, list[str]]) -> set[str]:
	"""Recursively generate phonetic spelling variations based on rules."""
	if not name:
		return {""}
	char = name[0]
	rest = name[1:]
	vars_rest = generate_english_spelling_variations(rest, rules)
	current: set[str] = set()
	for substitution in rules.get(char, [char]):
		for variant in vars_rest:
			current.add(f"{substitution}{variant}")
	return current


def generate_all_name_variations(input_name: str) -> dict[str, set[str]]:
	"""Generate base Marathi/English variants by transliteration rules."""
	if not input_name:
		return {"marathi": set(), "english": set()}

	base_english = ""
	base_devanagari = ""

	try:
		if is_devanagari(input_name):
			base_devanagari = input_name
			raw_base_english = transliterate(input_name, sanscript.DEVANAGARI, sanscript.ITRANS)
			base_english = clean_english_name(raw_base_english).lower()
		else:
			base_english = clean_english_name(input_name).lower()
			try:
				base_devanagari = transliterate(base_english, sanscript.ITRANS, sanscript.DEVANAGARI)
			except Exception:
				temp_name = base_english.replace("i", "ee").replace("oo", "U")
				base_devanagari = transliterate(temp_name, sanscript.ITRANS, sanscript.DEVANAGARI)
	except Exception as exc:
		logger.warning("Transliteration failed for '%s': %s", input_name, exc)
		# Fall back to returning the original name in whichever script it was provided
		if is_devanagari(input_name):
			return {"marathi": {input_name}, "english": set()}
		return {"marathi": set(), "english": {clean_english_name(input_name)}}

	logger.debug("Detected base names -> English: '%s', Marathi: '%s'", base_english, base_devanagari)

	marathi_variations: set[str] = {base_devanagari} if base_devanagari else set()
	rules_marathi = {
		"ी": "ि",
		"ि": "ी",
		"ू": "ु",
		"ु": "ू",
		"श": "स",
		"स": "श",
		"व": "व",
		"ब": "ब",
	}
	for _ in range(2):
		temp_variations: set[str] = set()
		for name in marathi_variations:
			for char, replacement in rules_marathi.items():
				if char in name:
					temp_variations.add(name.replace(char, replacement, 1))
		marathi_variations.update(temp_variations)

	english_variations: set[str] = set()
	for name in marathi_variations:
		try:
			raw_english = transliterate(name.replace("\u00A0", " "), sanscript.DEVANAGARI, sanscript.ITRANS)
			cleaned = clean_english_name(raw_english)
			if cleaned:
				english_variations.add(cleaned)
		except Exception:
			continue

	rules_english = {
		"i": ["i", "e"],
		"u": ["u", "o"],
		"v": ["v", "w"],
		"j": ["j", "z"],
	}
	for variant in generate_english_spelling_variations(base_english, rules_english):
		cleaned = clean_english_name(variant)
		if cleaned:
			english_variations.add(cleaned)

	return {
		"marathi": {v for v in marathi_variations if v},
		"english": {v for v in english_variations if v},
	}


def get_marathi_initial(word: str) -> str:
	if not word:
		return ""
	match = regex.match(r"\X", word)
	return match.group(0) if match else word[:1]


def get_english_initial(word: str) -> str:
	return word[0].lower() if word else ""


def add_initial_variations(name_dict: dict[str, set[str]]) -> dict[str, set[str]]:
	def generate_variants(name: str, *, is_marathi: bool) -> set[str]:
		parts = name.split()
		if len(parts) < 2:
			return set()
		first = parts[0]
		middle = parts[1] if len(parts) == 3 else ""
		last = parts[-1]

		if is_marathi:
			first_i = get_marathi_initial(first)
			middle_i = get_marathi_initial(middle) if middle else ""
			last_i = get_marathi_initial(last)
		else:
			first_i = get_english_initial(first)
			middle_i = get_english_initial(middle) if middle else ""
			last_i = get_english_initial(last)

		variants: set[str] = set()
		if middle:
			variants.add(f"{first_i} {middle_i} {last}")
			variants.add(f"{first_i} {middle_i}")
			variants.add(f"{first} {middle_i} {last}")
			variants.add(f"{first_i} {middle} {last}")
		else:
			variants.add(f"{first_i} {last}")
			variants.add(f"{first} {last_i}")
		return {v.strip() for v in variants if v.strip()}

	new_marathi = set(name_dict.get("marathi", []))
	for name in list(new_marathi):
		new_marathi |= generate_variants(name, is_marathi=True)

	new_english = set(name_dict.get("english", []))
	for name in list(new_english):
		new_english |= generate_variants(name, is_marathi=False)

	return {"marathi": new_marathi, "english": new_english}


def _generate_name_variations(name: str) -> dict:
	if not name:
		return {"english_variants": [], "marathi_variants": []}
	base_variations = generate_all_name_variations(name)
	with_initials = add_initial_variations(base_variations)
	english = sorted(with_initials.get("english", []))
	marathi = sorted(with_initials.get("marathi", []))
	return {"english_variants": english, "marathi_variants": marathi}


@app.get("/phonetics")
def phonetics(seed_name: str = Query(..., description="Name to generate phonetic/variation suggestions for")) -> dict:
	if not seed_name:
		raise HTTPException(status_code=400, detail="seed_name required")
	
	variations = _generate_name_variations(seed_name)
	
	return {
		"input": seed_name,
		"variations": {
			"english": variations.get("english_variants", []),
			"marathi": variations.get("marathi_variants", []),
		},
	}


@app.get("/search")
def search(
	pan: Optional[str] = Query(default=None),
	seed_name: Optional[str] = Query(default=None),
	limit: Optional[int] = Query(default=1000),
) -> dict:
	if not pan and not seed_name:
		raise HTTPException(status_code=400, detail="Provide either pan or seed_name")
	if not os.path.exists(DB_PATH):
		raise HTTPException(status_code=500, detail=f"DB not found at {DB_PATH}")
	con = duckdb.connect(DB_PATH, read_only=True)
	try:
		if pan:
			rows = _search_by_pan(con, pan)
		else:
			rows = _search_by_seed_name(con, seed_name or "")
		rows = rows[: (limit or len(rows))]
		# Fetch column names for transactions table
		cols = [r[1] for r in con.execute("PRAGMA table_info('transactions')").fetchall()]
		# Prepare scoring context
		idx_name = cols.index("name_norm") if "name_norm" in cols else -1
		idx_pan = cols.index("pan_upper") if "pan_upper" in cols else -1
		idx_year = cols.index("year") if "year" in cols else -1
		# Log if year column is missing (for debugging)
		if idx_year == -1:
			logger.warning("'year' column not found in transactions table schema. Available columns: %s", cols[:10])
		base_names: list[str] = []
		if pan:
			try:
				base_rows = _fetch_rows_for_pan(con, pan)
				base_names = _pick_canonical_names([(r[0],) for r in base_rows])
			except Exception:
				base_names = []
		elif seed_name:
			base_names = [nlp.normalize_name(seed_name or "")]
		pan_up = nlp.canonicalize_pan(pan) if pan else None

		data = []
		for row in rows:
			score = 0.0
			if pan and idx_pan >= 0 and row[idx_pan] and pan_up and row[idx_pan] == pan_up:
				score = 1.0
			if score < 1.0 and idx_name >= 0 and row[idx_name] and base_names:
				best = max((nlp.fuzzy_name_score(row[idx_name], b) for b in base_names), default=0)
				score = max(score, best / 100.0)
			item = dict(zip(cols, row))
			# drop internal helper columns from API response
			item.pop("pan_upper", None)
			item.pop("pan_raw", None)
			# Ensure year column is included (explicitly set if it exists in the row)
			# The dict(zip()) should already include it, but we explicitly ensure it's there
			if idx_year >= 0:
				if idx_year < len(row):
					item["year"] = row[idx_year]
				else:
					# Row doesn't have enough elements - this shouldn't happen but handle gracefully
					logger.warning("Row length mismatch: expected year at index %d but row has %d elements", idx_year, len(row))
					item["year"] = None
			# The year column is needed for age calculation on the client side
			item["match_score"] = round(float(score), 3)
			# Add highlight for seed_name searches: highlight ALL tokens in the matched field
			if seed_name:
				seed_norm_local = nlp.normalize_name(seed_name or "")
				tokens = [t for t in set((seed_norm_local or "").split(" ")) if len(t) >= 2]
				fields_to_check = [
					"buyer",
					"seller",
					"_name_primary",
					"_name_alt",
					"name_extracted",
				]
				hit_field = ""
				hit_snippet = ""
				hit_snippet_html = ""
				for f in fields_to_check:
					val = item.get(f)
					if not isinstance(val, str) or not val:
						continue
					low = val.lower()
					# Check that ALL tokens are present
					if all(t in low for t in tokens):
						# Build HTML by marking every token occurrence
						marked = val
						for t in sorted(tokens, key=len, reverse=True):
							try:
								pattern = re.compile(re.escape(t), re.IGNORECASE)
								marked = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", marked)
							except Exception:
								pass
						hit_field = f
						hit_snippet_html = marked
						# Plain variant with «» is not as useful for multi-token; omit
						break
				if hit_field:
					item["name_hit_field"] = hit_field
					item["name_hit_snippet"] = hit_snippet
					item["name_hit_snippet_html"] = hit_snippet_html

			# Add highlight for PAN searches: show where exact PAN appears
			if pan:
				pan_up_local = nlp.canonicalize_pan(pan)
				pan_fields = ["pan_numbers", "buyer", "seller"]
				p_hit_field = ""
				p_hit_html = ""
				for f in pan_fields:
					val = item.get(f)
					if not isinstance(val, str) or not val:
						continue
					text_up = val.upper()
					pos = text_up.find(pan_up_local)
					if pos >= 0:
						start = max(0, pos - 10)
						end = min(len(val), pos + len(pan_up_local) + 10)
						before = val[start:pos]
						mid = val[pos:pos+len(pan_up_local)]
						after = val[pos+len(pan_up_local):end]
						p_hit_field = f
						p_hit_html = f"{before}<mark>{mid}</mark>{after}"
						break
				if p_hit_field:
					item["pan_hit_field"] = p_hit_field
					item["pan_hit_snippet_html"] = p_hit_html
			data.append(item)
		
		# Calculate Property Type for each row
		# Get the entity name for comparison
		entity_name_for_comparison = None
		if pan:
			entity_name_for_comparison = pan_up
		elif seed_name:
			entity_name_for_comparison = nlp.normalize_name(seed_name or "")
		
		# Process each row to determine Property Type
		for item in data:
			property_type = "N/A"
			
			# Get the Name from this row (the entity we're searching for)
			row_name = item.get("Name") or item.get("name_extracted") or ""
			if not row_name and entity_name_for_comparison:
				row_name = entity_name_for_comparison
			
			# Extract names and PANs from Buyer and Seller columns
			buyer_text = str(item.get("buyer", "") or "")
			seller_text = str(item.get("seller", "") or "")
			
			buyer_names = _extract_names_from_marathi_string(buyer_text)
			seller_names = _extract_names_from_marathi_string(seller_text)
			buyer_pans = _extract_pan_numbers_from_text(buyer_text)
			seller_pans = _extract_pan_numbers_from_text(seller_text)
			
			# Check if name appears in buyer or seller
			is_buyer = False
			is_seller = False
			
			if row_name:
				# Normalize for comparison
				row_name_normalized = nlp.normalize_name(row_name) if hasattr(nlp, 'normalize_name') else row_name.upper()
				row_name_upper = row_name.upper()
				
				# Check buyer names
				for b_name in buyer_names:
					b_name_normalized = nlp.normalize_name(b_name) if hasattr(nlp, 'normalize_name') else b_name.upper()
					b_name_upper = b_name.upper()
					# Check for partial matches (name in buyer name or vice versa)
					if (row_name_normalized in b_name_normalized or b_name_normalized in row_name_normalized or
						row_name_upper in b_name_upper or b_name_upper in row_name_upper):
						is_buyer = True
						break
				
				# Check seller names
				for s_name in seller_names:
					s_name_normalized = nlp.normalize_name(s_name) if hasattr(nlp, 'normalize_name') else s_name.upper()
					s_name_upper = s_name.upper()
					# Check for partial matches (name in seller name or vice versa)
					if (row_name_normalized in s_name_normalized or s_name_normalized in row_name_normalized or
						row_name_upper in s_name_upper or s_name_upper in row_name_upper):
						is_seller = True
						break
			
			# Also check PAN numbers if searching by PAN
			if pan and pan_up:
				pan_up_normalized = pan_up.upper()
				# Check if search PAN appears in buyer PANs
				if pan_up_normalized in buyer_pans:
					is_buyer = True
				# Check if search PAN appears in seller PANs
				if pan_up_normalized in seller_pans:
					is_seller = True
			
			# Determine property type
			if is_buyer and not is_seller:
				property_type = "Bought"
			elif is_seller and not is_buyer:
				property_type = "Sold"
			elif is_buyer and is_seller:
				property_type = "Both"
			
			item["property_type"] = property_type
		
		return {"count": len(rows), "data": data}

	finally:
		con.close()


def _extract_person_name_llm(raw_name: str) -> Optional[str]:
	"""
	Use local LLM (OpenAI-compatible API via Ollama) to extract the main human person's name
	from a longer Marathi/English phrase. Returns a concise name or None on failure.
	"""
	if not raw_name:
		return None
	try:
		# Create custom httpx client to avoid proxies parameter issue
		http_client = httpx.Client(timeout=30.0)
		client = OpenAI(
			base_url="http://192.168.1.198:11434/v1",
			api_key="ollama",
			http_client=http_client
		)
		system_prompt = "Return JSON only. You extract concise person names."
		user_prompt = (
			"Extract only the main human person's name from the text.\n"
			"- Text may include roles (e.g., संचालक, तर्फे), organizations, or punctuation.\n"
			"- Preserve the script (Marathi remains Marathi, English remains English).\n"
			"- Respond ONLY as compact JSON: {\\\"name\\\": \\\"<NAME>\\\"}. No extra text.\n\n"
			f"Text: {raw_name}"
		)
		resp = client.chat.completions.create(
			model="gpt-oss:20b",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=0,
		)
		text = (resp.choices[0].message.content or "").strip()
		if not text:
			logger.warning("LLM returned empty content for extraction")
		# Parse JSON strictly; tolerate minor wrappers
		name_val = None
		try:
			obj = json.loads(text)
			name_val = obj.get("name")
		except Exception:
			# try to find a minimal JSON snippet in the text
			m = re.search(r"\{\s*\"name\"\s*:\s*\"(.+?)\"\s*\}", text)
			if m:
				name_val = m.group(1)
		if name_val:
			name_val = str(name_val).strip().strip('"\'\u201c\u201d')
			if 0 < len(name_val) <= 60:
				return name_val
		# Fallback heuristic for Marathi phrases: pick last 2 tokens that are not role/stop words
		stop = {"चे","चा","ची","चे","प्रा","ली","लि","अँड","एंड","तर्फे","संचालक","मुखत्यार","प्रा.","लि.","कंपनी","लिमिटेड","प्रा.लि."}
		words = re.split(r"\s+", raw_name.strip())
		filtered = [w for w in words if w and (re.sub(r"[\,;:\-]", "", w) not in stop)]
		if len(filtered) >= 1:
			cand = " ".join(filtered[-2:]) if len(filtered) >= 2 else filtered[-1]
			cand = cand.strip()
			if 0 < len(cand) <= 60:
				logger.info("LLM fallback heuristic used for extraction")
				return cand
		return None
	except Exception as e:
		logger.exception("LLM extraction failed: %s", e)
		return None


@app.get("/pan_meta")
def pan_meta(pan: str = Query(..., description="PAN to fetch metadata for")) -> dict:
	if not pan:
		raise HTTPException(status_code=400, detail="pan required")
	pan_id = nlp.canonicalize_pan(pan)
	if not pan_id:
		raise HTTPException(status_code=400, detail="invalid pan")
	base_url = "http://192.168.1.198:5000/get_by_id"
	url = f"{base_url}?" + urllib.parse.urlencode({"id": pan_id})
	try:
		with urllib.request.urlopen(url, timeout=5) as resp:
			payload = resp.read()
			try:
				data = json.loads(payload.decode("utf-8"))
			except Exception:
				data = {}
		# Normalize keys
		name = data.get("name") or data.get("Name") or data.get("full_name") or data.get("fullName")
		age = data.get("age") if "age" in data else data.get("Age") if "Age" in data else data.get("person_age")
		# Coerce simple types
		if name is not None and not isinstance(name, str):
			name = str(name)
		if age is not None and not isinstance(age, (int, float)):
			try:
				age_num = int(str(age))
				age = age_num
			except Exception:
				pass
		# Return raw name without LLM extraction
		extracted = None
		final_name = name
		extraction_error = None
		return {"pan": pan_id, "name": final_name, "extracted_name": extracted, "raw_name": name, "age": age, "raw": data, "extraction_error": extraction_error}
	except Exception as e:
		logger.exception("PAN meta upstream error for %s: %s", pan_id, e)
		raise HTTPException(status_code=502, detail=f"upstream error: {e}")


@app.get("/llm_test")
def llm_test(text: str = Query(..., description="Sample text to extract name from")) -> dict:
	"""Simple endpoint to verify LLM extraction end-to-end."""
	try:
		extracted = _extract_person_name_llm(text)
		return {"input": text, "extracted": extracted}
	except Exception as e:
		logger.exception("LLM test failed: %s", e)
		return {"input": text, "extracted": None}


# -------------------- PDF Generation with Devanagari Support --------------------

def _register_devanagari_fonts():
	"""Register Devanagari-supporting fonts if available. Caches result globally."""
	global _DEVANAGARI_FONTS
	
	if _DEVANAGARI_FONTS is not None:
		return _DEVANAGARI_FONTS
	
	font_paths = [
		# macOS font locations (multiple possible paths)
		# Noto Sans Devanagari (TTF)
		"/System/Library/Fonts/Supplemental/NotoSansDevanagari-Regular.ttf",
		"/Library/Fonts/NotoSansDevanagari-Regular.ttf",
		"~/Library/Fonts/NotoSansDevanagari-Regular.ttf",
		"/System/Library/Fonts/Supplemental/NotoSansDevanagari-Bold.ttf",
		"/Library/Fonts/NotoSansDevanagari-Bold.ttf",
		"~/Library/Fonts/NotoSansDevanagari-Bold.ttf",
		# macOS built-in Devanagari fonts (TTC - TrueType Collection)
		"/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
		"/System/Library/Fonts/Supplemental/ITFDevanagari.ttc",
		"/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
		# Linux font locations
		"/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
		"/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
		"/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
		"/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf",
		# Windows font locations
		"C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
		"C:/Windows/Fonts/NotoSansDevanagari-Bold.ttf",
		# Alternative Unicode fonts
		"/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
		"/Library/Fonts/Arial Unicode.ttf",
		"C:/Windows/Fonts/ARIALUNI.TTF",
	]
	
	# Expand home directory paths
	expanded_paths = []
	for path in font_paths:
		if path.startswith("~"):
			expanded_paths.append(os.path.expanduser(path))
		else:
			expanded_paths.append(path)
	font_paths = expanded_paths
	
	# Also try to find fonts dynamically (for macOS .ttc files)
	try:
		import glob
		# Search for any Devanagari font files
		search_paths = [
			"/System/Library/Fonts/Supplemental/*Devanagari*",
			"/Library/Fonts/*Devanagari*",
			os.path.expanduser("~/Library/Fonts/*Devanagari*"),
		]
		for pattern in search_paths:
			for font_file in glob.glob(pattern):
				if font_file not in font_paths and (font_file.endswith('.ttf') or font_file.endswith('.ttc')):
					font_paths.append(font_file)
	except Exception as e:
		logger.debug(f"Could not search for fonts dynamically: {e}")
	
	registered = {"regular": None, "bold": None}
	
	for font_path in font_paths:
		if os.path.exists(font_path):
			try:
				# Determine if this is a bold font
				is_bold = "Bold" in font_path or "bold" in font_path.lower() or "ITF" in font_path
				
				# For TTC files, we might need to specify subfont index (0 for regular, 1 for bold)
				# But TTFont can usually handle TTC files automatically
				if is_bold and not registered["bold"]:
					pdfmetrics.registerFont(TTFont("Devanagari-Bold", font_path))
					registered["bold"] = "Devanagari-Bold"
					logger.info(f"Registered Devanagari bold font: {font_path}")
				elif not is_bold and not registered["regular"]:
					pdfmetrics.registerFont(TTFont("Devanagari", font_path))
					registered["regular"] = "Devanagari"
					logger.info(f"Registered Devanagari regular font: {font_path}")
				
				# If we have both fonts, we can stop searching
				if registered["regular"] and registered["bold"]:
					break
			except Exception as e:
				logger.warning(f"Failed to register font {font_path}: {e}")
				continue
	
	# If no Devanagari font found, use default (will show garbled text for Devanagari)
	if not registered["regular"]:
		logger.warning("No Devanagari font found. Devanagari text may not render correctly.")
		logger.warning("To enable Devanagari support, install Noto Sans Devanagari font:")
		logger.warning("  macOS: Font is usually pre-installed")
		logger.warning("  Linux: sudo apt-get install fonts-noto-core")
		logger.warning("  Windows: Download from https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari")
		registered["regular"] = "Helvetica"
		registered["bold"] = "Helvetica-Bold"
	
	_DEVANAGARI_FONTS = registered
	return registered


def _strip_html(text):
	"""Remove HTML tags from text."""
	if not text:
		return ""
	return re.sub(r"<[^>]+>", "", str(text))


def _extract_names_from_marathi_string(text: str) -> List[str]:
	"""Extract names from Marathi/English formatted text.
	
	Handles formats like:
	- Marathi: '1): नाव:-Name वयः-Age ... 2): नाव:-Name'
	- English: '1) Name: . Name Age: 39 ... 2) Name: Name'
	"""
	if not text:
		return []
	names = []
	text_str = str(text)
	
	# Pattern 1: Marathi format - "नाव:-" or "नावः-" followed by name
	marathi_pattern = r'नाव[ः-]-\s*([^;]+?)(?=\s*वयः-|\s*पत्ता:-|\s*पॅन नं:-|\s*\d\):|\s*नाव[ः-]|$)'
	marathi_matches = re.findall(marathi_pattern, text_str, re.IGNORECASE)
	for match in marathi_matches:
		name = match.strip()
		if name:
			names.append(name)
	
	# Pattern 2: English format - "Name: . Name" or "Name: Name"
	# This handles cases like "Name: . Tarun Prakash Lalwani Age: 39"
	english_pattern = r'(?:^|\d\)\s*)Name:\s*\.?\s*([^;]+?)(?=\s*Age:|\s*Address:|\s*PAN:|\s*\d\):|$)'
	english_matches = re.findall(english_pattern, text_str, re.IGNORECASE)
	for match in english_matches:
		name = match.strip()
		if name and name not in names:  # Avoid duplicates
			names.append(name)
	
	return names


def _extract_pan_numbers_from_text(text: str) -> List[str]:
	"""Extract PAN numbers from text (both Marathi and English formats)."""
	if not text:
		return []
	pans = []
	text_str = str(text)
	
	# Pattern to match PAN numbers (10 character alphanumeric)
	# Handles formats like "पॅन नं:-AAACO8053A" or "PAN: AAACO8053A"
	pan_pattern = r'(?:पॅन नं[ः-]|PAN[:\s]+)([A-Z]{5}\d{4}[A-Z])'
	pan_matches = re.findall(pan_pattern, text_str, re.IGNORECASE)
	for match in pan_matches:
		pan = match.strip().upper()
		if pan and pan not in pans:
			pans.append(pan)
	
	# Also look for standalone PAN patterns in the text
	standalone_pan_pattern = r'\b([A-Z]{5}\d{4}[A-Z])\b'
	standalone_matches = re.findall(standalone_pan_pattern, text_str)
	for match in standalone_matches:
		pan = match.strip().upper()
		if pan and pan not in pans:
			pans.append(pan)
	
	return pans


def _generate_pdf(data_rows: List[dict], cols_display: List[str], col_map: dict, search_pan: Optional[str] = None) -> BytesIO:
	"""Generate PDF with header, footer, and Devanagari support."""
	if not data_rows:
		raise ValueError("No data rows provided for PDF generation")
	
	buffer = BytesIO()
	# Use landscape orientation
	page_size = landscape(A4)
	# Increased top margin to ensure logo is fully visible, increased bottom margin for footer image
	doc = SimpleDocTemplate(buffer, pagesize=page_size, topMargin=30*mm, bottomMargin=30*mm, leftMargin=15*mm, rightMargin=15*mm)
	
	# Register Devanagari fonts
	fonts = _register_devanagari_fonts()
	devanagari_font = fonts["regular"]
	devanagari_bold = fonts["bold"]
	
	# Verify fonts are registered (fallback to Helvetica if not)
	# Standard reportlab fonts: Helvetica, Times-Roman, Courier (and their variants)
	standard_fonts = {"Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
	                  "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
	                  "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"}
	
	# Check if custom fonts are actually registered
	# If font is not a standard font, verify it's registered
	if devanagari_font not in standard_fonts:
		try:
			# Try to get font info - this will raise if font doesn't exist
			pdfmetrics.getFont(devanagari_font)
		except (KeyError, AttributeError):
			logger.warning(f"Custom font {devanagari_font} not properly registered, using Helvetica")
			devanagari_font = "Helvetica"
	
	if devanagari_bold not in standard_fonts:
		try:
			pdfmetrics.getFont(devanagari_bold)
		except (KeyError, AttributeError):
			logger.warning(f"Custom font {devanagari_bold} not properly registered, using Helvetica-Bold")
			devanagari_bold = "Helvetica-Bold"
	
	# Create styles
	styles = getSampleStyleSheet()
	
	# Professional color scheme - Modern Navy & Gold
	primary_color = colors.HexColor("#1E3A5F")  # Rich navy blue
	primary_dark = colors.HexColor("#152238")  # Darker navy
	accent_color = colors.HexColor("#D4AF37")  # Elegant gold
	text_dark = colors.HexColor("#1A1A1A")  # Dark text
	text_medium = colors.HexColor("#424242")  # Medium text
	text_light = colors.HexColor("#6B7280")  # Light text
	bg_light = colors.HexColor("#F8F9FA")  # Very light background
	bg_blue_light = colors.HexColor("#E8F0F8")  # Light blue background
	border_color = colors.HexColor("#D1D5DB")  # Soft border color
	
	# Custom styles - use safe fonts with fallback
	try:
		title_style = ParagraphStyle(
			"CustomTitle",
			parent=styles["Heading1"],
			fontSize=16,
			fontName=devanagari_bold,
			textColor=primary_dark,
			alignment=TA_CENTER,
			spaceAfter=8,
		)
	except Exception:
		title_style = ParagraphStyle(
			"CustomTitle",
			parent=styles["Heading1"],
			fontSize=16,
			fontName="Helvetica-Bold",
			textColor=primary_dark,
			alignment=TA_CENTER,
			spaceAfter=8,
		)
	
	try:
		heading_style = ParagraphStyle(
			"CustomHeading",
			parent=styles["Heading2"],
			fontSize=13,
			fontName=devanagari_bold,
			textColor=primary_dark,
			alignment=TA_LEFT,
			spaceAfter=6,
		)
	except Exception:
		heading_style = ParagraphStyle(
			"CustomHeading",
			parent=styles["Heading2"],
			fontSize=13,
			fontName="Helvetica-Bold",
			textColor=primary_dark,
			alignment=TA_LEFT,
			spaceAfter=6,
		)
	
		try:
			normal_style = ParagraphStyle(
				"CustomNormal",
				parent=styles["Normal"],
				fontSize=9,
				fontName=devanagari_font,
				textColor=text_dark,
				alignment=TA_LEFT,
			)
		except Exception:
			normal_style = ParagraphStyle(
				"CustomNormal",
				parent=styles["Normal"],
				fontSize=9,
				fontName="Helvetica",
				textColor=text_dark,
				alignment=TA_LEFT,
			)
		
		try:
			small_style = ParagraphStyle(
				"CustomSmall",
				parent=styles["Normal"],
				fontSize=8,
				fontName=devanagari_font,
				textColor=text_light,
				alignment=TA_LEFT,
			)
		except Exception:
			small_style = ParagraphStyle(
				"CustomSmall",
				parent=styles["Normal"],
				fontSize=8,
				fontName="Helvetica",
				textColor=text_light,
				alignment=TA_LEFT,
			)
	
	# Story (content) list
	story = []
	
	# Helper function to create paragraph with Devanagari support
	def para(text, style=None):
		if style is None:
			style = normal_style
		if not text:
			return Paragraph("", style)
		text = _strip_html(str(text))
		# Escape special characters for reportlab
		text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
		return Paragraph(text, style)
	
	# PAGE 1: Header and Summary
	# Logo is added via canvas, so we start with the main title
	# Add some top spacing to account for logo
	story.append(Spacer(1, 8*mm))
	
	# Main title with decorative underline
	title_para = Paragraph("Property Summary Report for Financial Creditor", title_style)
	story.append(title_para)
	story.append(Spacer(1, 4*mm))
	
	# Entity name - use PAN number if provided, otherwise fallback to name
	if search_pan:
		# Use the searched PAN number (canonicalized)
		entity_display = nlp.canonicalize_pan(search_pan) or search_pan.upper()
	else:
		# Fallback to name or PAN from data
		entity_display = data_rows[0].get("name_extracted") or data_rows[0].get("Name") or data_rows[0].get("pan_numbers") or "N/A"
	
	# Entity name with badge-like styling using a table
	try:
		entity_style = ParagraphStyle("EntityName", parent=styles["Normal"], fontSize=15, fontName=devanagari_bold, 
			textColor=colors.white, alignment=TA_CENTER)
	except Exception:
		entity_style = ParagraphStyle("EntityName", parent=styles["Normal"], fontSize=15, fontName="Helvetica-Bold", 
			textColor=colors.white, alignment=TA_CENTER)
	
	entity_para = Paragraph(entity_display.upper(), entity_style)
	entity_table = Table([[entity_para]], colWidths=[page_size[0] - 30*mm])
	entity_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, -1), primary_dark),
		("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
		("ALIGN", (0, 0), (-1, -1), "CENTER"),
		("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
		("TOPPADDING", (0, 0), (-1, -1), 8),
		("BOTTOMPADDING", (0, 0), (-1, -1), 8),
		("LEFTPADDING", (0, 0), (-1, -1), 12),
		("RIGHTPADDING", (0, 0), (-1, -1), 12),
		("LINEBELOW", (0, 0), (-1, -1), 2, accent_color),  # Accent line below
	]))
	story.append(entity_table)
	story.append(Spacer(1, 10*mm))
	
	# "To:" section with styled box
	to_style = ParagraphStyle("ToSection", parent=styles["Normal"], fontSize=9, fontName=devanagari_font, 
		textColor=text_medium, leftIndent=0, spaceAfter=2)
	try:
		to_heading_style = ParagraphStyle("ToHeading", parent=styles["Normal"], fontSize=10, fontName=devanagari_bold, 
			textColor=primary_dark, leftIndent=0, spaceAfter=4)
	except Exception:
		to_heading_style = ParagraphStyle("ToHeading", parent=styles["Normal"], fontSize=10, fontName="Helvetica-Bold", 
			textColor=primary_dark, leftIndent=0, spaceAfter=4)
	
	# Create a styled "To:" section with border
	to_data = [
		[Paragraph("<b>To,</b>", to_heading_style)],
		[Paragraph("The DGM,", to_style)],
		[Paragraph("legal & recovery department,", to_style)],
		[Paragraph("Abhyudaya co op bank ltd.,", to_style)],
		[Paragraph("parel", to_style)],
		[Paragraph("MOB NO. 8591948712/8169452713", to_style)],
	]
	to_table = Table(to_data, colWidths=[120*mm])
	to_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, -1), bg_light),
		("TEXTCOLOR", (0, 0), (-1, -1), text_medium),
		("GRID", (0, 0), (-1, -1), 1, border_color),
		("VALIGN", (0, 0), (-1, -1), "TOP"),
		("LEFTPADDING", (0, 0), (-1, -1), 8),
		("RIGHTPADDING", (0, 0), (-1, -1), 8),
		("TOPPADDING", (0, 0), (-1, -1), 6),
		("BOTTOMPADDING", (0, 0), (-1, -1), 6),
		("LINEBELOW", (0, 0), (-1, 0), 2, primary_color),  # Blue line under "To,"
	]))
	story.append(to_table)
	story.append(Spacer(1, 10*mm))
	
	# Table title with accent bar
	table_title_para = Paragraph("1) Details : Transaction Summary", heading_style)
	story.append(table_title_para)
	story.append(Spacer(1, 6*mm))
	
	# Create table - use Devanagari font if registered, otherwise fallback
	# If devanagari_font is NOT a standard font, it means we successfully registered a custom font
	standard_fonts_set = {"Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
	                      "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
	                      "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"}
	
	# Use Devanagari font if it's a custom registered font (not in standard fonts)
	if devanagari_font not in standard_fonts_set:
		table_font = devanagari_font  # Use the registered Devanagari font
		logger.info(f"Using Devanagari font for tables: {table_font}")
	else:
		table_font = "Helvetica"  # Fallback to Helvetica
		logger.warning("Devanagari font not available, using Helvetica (Marathi text may not render correctly)")
	
	if devanagari_bold not in standard_fonts_set:
		table_bold = devanagari_bold  # Use the registered Devanagari bold font
		logger.info(f"Using Devanagari bold font for tables: {table_bold}")
	else:
		table_bold = "Helvetica-Bold"  # Fallback to Helvetica-Bold
	
	# Summary table - use Paragraph objects for Unicode support
	summary_headers = ["Sr"] + cols_display
	summary_data = []
	
	# Helper to create Paragraph for table cells with proper Unicode/Devanagari support
	def cell_para(text, is_header=False):
		if text is None:
			text = ""
		text = _strip_html(str(text))
		# Escape XML special chars for reportlab Paragraph
		text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
		# Ensure text is properly encoded as UTF-8 string
		if isinstance(text, bytes):
			text = text.decode('utf-8', errors='ignore')
		# Use appropriate style with Devanagari font
		# Note: ParagraphStyle doesn't have an encoding parameter - reportlab handles UTF-8 automatically
		try:
			if is_header:
				style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontSize=9, fontName=table_bold, textColor=colors.white)
			else:
				style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, fontName=table_font, textColor=text_dark)
		except Exception:
			# Fallback if style creation fails
			if is_header:
				style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=colors.white)
			else:
				style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, fontName="Helvetica", textColor=text_dark)
		return Paragraph(text, style)
	
	# Convert headers to Paragraphs
	header_paras = [cell_para(h, is_header=True) for h in summary_headers]
	
	for idx, row in enumerate(data_rows):
		row_data = [cell_para(str(idx + 1))]  # Serial number
		for header in cols_display:
			key = col_map.get(header, header)
			val = row.get(key, "")
			# Limit very long values to prevent PDF issues
			if val is not None:
				val_str = str(val)
				if len(val_str) > 500:
					val_str = val_str[:500] + "..."
				row_data.append(cell_para(val_str))
			else:
				row_data.append(cell_para(""))
		summary_data.append(row_data)
	
	# Calculate column widths for landscape (more space available)
	# Serial number: 20mm, rest distributed evenly
	available_width = page_size[0] - 30*mm  # Total width minus margins
	serial_width = 20*mm
	remaining_width = available_width - serial_width
	col_widths = [serial_width] + [remaining_width / len(cols_display)] * len(cols_display)
	
	table = Table([header_paras] + summary_data, colWidths=col_widths)
	table.setStyle(TableStyle([
		# Header row with gradient-like blue background
		("BACKGROUND", (0, 0), (-1, 0), primary_dark),
		("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
		("ALIGN", (0, 0), (-1, 0), "LEFT"),
		("FONTNAME", (0, 0), (-1, 0), table_bold),
		("FONTSIZE", (0, 0), (-1, 0), 9),
		("BOTTOMPADDING", (0, 0), (-1, 0), 8),
		("TOPPADDING", (0, 0), (-1, 0), 8),
		("LEFTPADDING", (0, 0), (-1, 0), 6),
		("RIGHTPADDING", (0, 0), (-1, 0), 6),
		# Accent line below header
		("LINEBELOW", (0, 0), (-1, 0), 2, accent_color),
		# Data rows
		("BACKGROUND", (0, 1), (-1, -1), colors.white),
		("TEXTCOLOR", (0, 1), (-1, -1), text_dark),
		("ALIGN", (0, 1), (0, -1), "CENTER"),  # Serial number centered
		("ALIGN", (1, 1), (-1, -1), "LEFT"),
		("FONTNAME", (0, 1), (-1, -1), table_font),
		("FONTSIZE", (0, 1), (-1, -1), 8),
		("GRID", (0, 0), (-1, -1), 0.5, border_color),
		("VALIGN", (0, 0), (-1, -1), "TOP"),
		("LEFTPADDING", (0, 1), (-1, -1), 6),
		("RIGHTPADDING", (0, 1), (-1, -1), 6),
		("TOPPADDING", (0, 1), (-1, -1), 5),
		("BOTTOMPADDING", (0, 1), (-1, -1), 5),
		# Alternating row colors with subtle difference
		("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
		# Hover-like effect on first column (serial number)
		("BACKGROUND", (0, 1), (0, -1), bg_blue_light),
	]))
	
	story.append(table)
	# Footer is now added via canvas in _add_first_page_header_footer
	
	# Detailed transaction pages
	for idx, row in enumerate(data_rows):
		story.append(PageBreak())
		
		# Page header with styled background
		try:
			page_header = ParagraphStyle("PageHeader", parent=styles["Heading1"], fontSize=18, fontName=devanagari_bold, 
				textColor=colors.white, alignment=TA_LEFT, backColor=primary_dark, 
				leftIndent=12, rightIndent=12, spaceBefore=0, spaceAfter=0)
		except Exception:
			page_header = ParagraphStyle("PageHeader", parent=styles["Heading1"], fontSize=18, fontName="Helvetica-Bold", 
				textColor=colors.white, alignment=TA_LEFT, backColor=primary_dark, 
				leftIndent=12, rightIndent=12, spaceBefore=0, spaceAfter=0)
		
		header_table = Table([[Paragraph("&nbsp;Transaction Search Report&nbsp;", page_header)]], colWidths=[page_size[0] - 30*mm])
		header_table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, -1), primary_dark),
			("LINEBELOW", (0, 0), (-1, -1), 3, accent_color),
			("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
			("TOPPADDING", (0, 0), (-1, -1), 10),
			("BOTTOMPADDING", (0, 0), (-1, -1), 10),
		]))
		story.append(header_table)
		story.append(Spacer(1, 6*mm))
		
		date_str = datetime.now().strftime("%B %d, %Y")
		try:
			date_style = ParagraphStyle("Date", parent=styles["Normal"], fontSize=9, fontName=devanagari_font, 
				textColor=text_light, alignment=TA_RIGHT)
		except Exception:
			date_style = ParagraphStyle("Date", parent=styles["Normal"], fontSize=9, fontName="Helvetica", 
				textColor=text_light, alignment=TA_RIGHT)
		date_para = Paragraph(f"Generated on: {date_str}", date_style)
		story.append(date_para)
		story.append(Spacer(1, 8*mm))
		
		# Transaction title with badge
		try:
			trans_heading = ParagraphStyle("TransHeading", parent=styles["Heading2"], fontSize=14, fontName=devanagari_bold, 
				textColor=primary_dark, alignment=TA_LEFT, spaceAfter=8)
		except Exception:
			trans_heading = ParagraphStyle("TransHeading", parent=styles["Heading2"], fontSize=14, fontName="Helvetica-Bold", 
				textColor=primary_dark, alignment=TA_LEFT, spaceAfter=8)
		story.append(Paragraph(f"Transaction {idx + 1} of {len(data_rows)}", trans_heading))
		story.append(Spacer(1, 6*mm))
		
		# Get all columns for this row
		all_cols = [k for k in row.keys() if k not in ["pan_upper", "pan_raw", "match_score", "name_hit_field", "name_hit_snippet", "name_hit_snippet_html", "pan_hit_field", "pan_hit_snippet_html"]]
		all_cols.sort()
		
		# Create detail table - use Paragraphs for Unicode support
		detail_data = []
		for col in all_cols:
			val = row.get(col, "")
			if val is not None:
				val_str = _strip_html(str(val))
				if len(val_str) > 300:  # More space in landscape
					val_str = val_str[:300] + "..."
			else:
				val_str = ""
			# Escape XML and ensure UTF-8 encoding
			val_str = val_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
			if isinstance(val_str, bytes):
				val_str = val_str.decode('utf-8', errors='ignore')
			# Create Paragraphs with Devanagari font support
			# Note: ParagraphStyle doesn't have an encoding parameter - reportlab handles UTF-8 automatically
			try:
				col_para = Paragraph(str(col), ParagraphStyle("DetailLabel", parent=styles["Normal"], fontSize=9, fontName=table_bold, textColor=colors.white))
				val_para = Paragraph(val_str, ParagraphStyle("DetailValue", parent=styles["Normal"], fontSize=8, fontName=table_font, textColor=text_dark))
			except Exception:
				# Fallback if style creation fails
				col_para = Paragraph(str(col), ParagraphStyle("DetailLabel", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=colors.white))
				val_para = Paragraph(val_str, ParagraphStyle("DetailValue", parent=styles["Normal"], fontSize=8, fontName="Helvetica", textColor=text_dark))
			detail_data.append([col_para, val_para])
		
		# Use more width in landscape mode
		detail_table = Table(detail_data, colWidths=[90*mm, None])
		detail_table.setStyle(TableStyle([
			# Label column with blue background
			("BACKGROUND", (0, 0), (0, -1), primary_dark),
			("TEXTCOLOR", (0, 0), (0, -1), colors.white),
			("FONTNAME", (0, 0), (0, -1), table_bold),
			("FONTSIZE", (0, 0), (0, -1), 9),
			("LEFTPADDING", (0, 0), (0, -1), 8),
			("RIGHTPADDING", (0, 0), (0, -1), 8),
			# Value column
			("BACKGROUND", (1, 0), (1, -1), colors.white),
			("TEXTCOLOR", (1, 0), (1, -1), text_dark),
			("FONTNAME", (1, 0), (1, -1), table_font),
			("FONTSIZE", (1, 0), (1, -1), 8),
			("LEFTPADDING", (1, 0), (1, -1), 8),
			("RIGHTPADDING", (1, 0), (1, -1), 8),
			# Grid with better borders
			("GRID", (0, 0), (-1, -1), 0.5, border_color),
			("LINEAFTER", (0, 0), (0, -1), 1, primary_color),  # Blue line after label column
			("VALIGN", (0, 0), (-1, -1), "TOP"),
			("TOPPADDING", (0, 0), (-1, -1), 6),
			("BOTTOMPADDING", (0, 0), (-1, -1), 6),
			# Alternating rows for value column
			("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, bg_light]),
		]))
		
		story.append(detail_table)
	
	# Build PDF with error handling
	# Get logo and footer paths
	logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "logo.png")
	footer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "footer.png")
	
	if not os.path.exists(logo_path):
		logo_path = None
		logger.warning("Logo not found at %s", logo_path)
	if not os.path.exists(footer_path):
		footer_path = None
		logger.warning("Footer image not found at %s", footer_path)
	
	try:
		doc.build(story, 
			onFirstPage=lambda canvas, doc: _add_first_page_header_footer(canvas, doc, logo_path, footer_path),
			onLaterPages=lambda canvas, doc: _add_page_header_footer(canvas, doc, logo_path, footer_path))
	except Exception as e:
		logger.exception("Error building PDF: %s", e)
		raise ValueError(f"Failed to build PDF: {str(e)}")
	
	buffer.seek(0)
	return buffer


def _add_first_page_header_footer(canvas, doc, logo_path=None, footer_path=None):
	"""Add header (logo) and footer (image) to first page with visual enhancements."""
	canvas.saveState()
	
	# Get page size (landscape A4)
	page_width = landscape(A4)[0]
	page_height = landscape(A4)[1]
	
	# Add decorative top bar (positioned at top margin)
	primary_color = colors.HexColor("#1E3A5F")  # Rich navy blue
	accent_color = colors.HexColor("#D4AF37")  # Elegant gold
	top_bar_y = page_height - 30*mm  # At top margin
	canvas.setFillColor(primary_color)
	canvas.rect(0, top_bar_y, page_width, 4*mm, fill=1, stroke=0)
	canvas.setFillColor(accent_color)
	canvas.rect(0, top_bar_y, page_width, 1.2*mm, fill=1, stroke=0)
	
	# Add logo at top right (with proper margin) - larger size
	if logo_path and os.path.exists(logo_path):
		try:
			logo_width = 60*mm  # Increased from 40mm
			logo_height = 22*mm  # Increased from 15mm (maintains aspect ratio approximately)
			# Position: 15mm from right edge, 15mm from top edge (accounting for 30mm top margin)
			logo_x = page_width - 15*mm - logo_width
			logo_y = page_height - 15*mm - logo_height
			canvas.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True)
		except Exception as e:
			logger.warning(f"Failed to add logo: {e}")
	
	# Footer image at bottom of page with decorative bar above
	if footer_path and os.path.exists(footer_path):
		try:
			# Add decorative bar above footer
			canvas.setFillColor(primary_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 2*mm, fill=1, stroke=0)
			canvas.setFillColor(accent_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 0.5*mm, fill=1, stroke=0)
			
			# Footer image dimensions - full width minus margins, auto height
			footer_width = page_width - 30*mm  # Full width minus left/right margins
			footer_height = 20*mm  # Fixed height for footer
			# Position: 15mm from left, 15mm from bottom (accounting for 30mm bottom margin)
			footer_x = 15*mm
			footer_y = 15*mm
			canvas.drawImage(footer_path, footer_x, footer_y, width=footer_width, height=footer_height, preserveAspectRatio=True)
		except Exception as e:
			logger.warning(f"Failed to add footer image: {e}")
			# Fallback to text footer if image fails
			canvas.setFont("Helvetica", 8)
			canvas.setFillColor(colors.HexColor("#424242"))  # Medium text color
			canvas.drawString(15*mm, 20*mm, "B1-001, U/B Floor, Boomerang Chandivali Farm Road, Powai, Andheri East, Mumbai, Maharashtra 400072")
			canvas.setFillColor(colors.HexColor("#1E3A5F"))  # Rich navy blue
			canvas.drawRightString(page_width - 15*mm, 20*mm, "www.real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, 16*mm, "info@real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, 12*mm, "+91 88284 05969")
	
	canvas.restoreState()


def _add_page_header_footer(canvas, doc, logo_path=None, footer_path=None):
	"""Add header (logo) and footer (image) to subsequent pages with visual enhancements."""
	canvas.saveState()
	
	# Get page size (landscape A4)
	page_width = landscape(A4)[0]
	page_height = landscape(A4)[1]
	
	# Add decorative top bar (positioned at top margin)
	primary_color = colors.HexColor("#1E3A5F")  # Rich navy blue
	accent_color = colors.HexColor("#D4AF37")  # Elegant gold
	top_bar_y = page_height - 30*mm  # At top margin
	canvas.setFillColor(primary_color)
	canvas.rect(0, top_bar_y, page_width, 4*mm, fill=1, stroke=0)
	canvas.setFillColor(accent_color)
	canvas.rect(0, top_bar_y, page_width, 1.2*mm, fill=1, stroke=0)
	
	# Add logo at top right (with proper margin) - larger size
	if logo_path and os.path.exists(logo_path):
		try:
			logo_width = 60*mm  # Increased from 40mm
			logo_height = 22*mm  # Increased from 15mm (maintains aspect ratio approximately)
			# Position: 15mm from right edge, 15mm from top edge (accounting for 30mm top margin)
			logo_x = page_width - 15*mm - logo_width
			logo_y = page_height - 15*mm - logo_height
			canvas.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True)
		except Exception as e:
			logger.warning(f"Failed to add logo: {e}")
	
	# Footer image at bottom of page with decorative bar above
	if footer_path and os.path.exists(footer_path):
		try:
			# Add decorative bar above footer
			canvas.setFillColor(primary_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 2*mm, fill=1, stroke=0)
			canvas.setFillColor(accent_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 0.5*mm, fill=1, stroke=0)
			
			# Footer image dimensions - full width minus margins, auto height
			footer_width = page_width - 30*mm  # Full width minus left/right margins
			footer_height = 20*mm  # Fixed height for footer
			# Position: 15mm from left, 15mm from bottom (accounting for 30mm bottom margin)
			footer_x = 15*mm
			footer_y = 15*mm
			canvas.drawImage(footer_path, footer_x, footer_y, width=footer_width, height=footer_height, preserveAspectRatio=True)
			
			# Add page number on top of footer image (centered) with styled background
			page_num = canvas.getPageNumber()
			canvas.setFont("Helvetica-Bold", 10)
			canvas.setFillColor(primary_color)
			canvas.drawCentredString(page_width / 2, footer_y + footer_height + 3*mm, f"Page {page_num}")
		except Exception as e:
			logger.warning(f"Failed to add footer image: {e}")
			# Fallback to text footer if image fails
			footer_y = 25*mm
			canvas.setFont("Helvetica-Bold", 10)
			canvas.setFillColor(primary_color)
			page_num = canvas.getPageNumber()
			canvas.drawCentredString(page_width / 2, footer_y - 5*mm, f"Page {page_num}")
			canvas.setFont("Helvetica", 8)
			canvas.setFillColor(colors.HexColor("#424242"))  # Medium text color
			canvas.drawString(15*mm, footer_y - 5*mm, "B1-001, U/B Floor, Boomerang Chandivali Farm Road, Powai, Andheri East, Mumbai, Maharashtra 400072")
			canvas.setFillColor(colors.HexColor("#1E3A5F"))  # Rich navy blue
			canvas.drawRightString(page_width - 15*mm, footer_y - 5*mm, "www.real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, footer_y - 9*mm, "info@real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, footer_y - 13*mm, "+91 88284 05969")
	
	canvas.restoreState()


@app.get("/export_pdf")
def export_pdf(
	pan: Optional[str] = Query(default=None),
	seed_name: Optional[str] = Query(default=None),
	limit: Optional[int] = Query(default=1000),
	threshold: Optional[float] = Query(default=0.75),
) -> Response:
	"""Export search results as PDF with Devanagari support."""
	if not pan and not seed_name:
		raise HTTPException(status_code=400, detail="Provide either pan or seed_name")
	if not os.path.exists(DB_PATH):
		raise HTTPException(status_code=500, detail=f"DB not found at {DB_PATH}")
	
	con = duckdb.connect(DB_PATH, read_only=True)
	try:
		if pan:
			rows = _search_by_pan(con, pan)
		else:
			rows = _search_by_seed_name(con, seed_name or "")
		
		rows = rows[: (limit or len(rows))]
		cols = [r[1] for r in con.execute("PRAGMA table_info('transactions')").fetchall()]
		
		# Prepare data similar to /search endpoint
		idx_name = cols.index("name_norm") if "name_norm" in cols else -1
		idx_pan = cols.index("pan_upper") if "pan_upper" in cols else -1
		idx_year = cols.index("year") if "year" in cols else -1
		
		base_names: list[str] = []
		if pan:
			try:
				base_rows = _fetch_rows_for_pan(con, pan)
				base_names = _pick_canonical_names([(r[0],) for r in base_rows])
			except Exception:
				base_names = []
		elif seed_name:
			base_names = [nlp.normalize_name(seed_name or "")]
		
		pan_up = nlp.canonicalize_pan(pan) if pan else None
		
		data = []
		for row in rows:
			score = 0.0
			if pan and idx_pan >= 0 and row[idx_pan] and pan_up and row[idx_pan] == pan_up:
				score = 1.0
			if score < 1.0 and idx_name >= 0 and row[idx_name] and base_names:
				best = max((nlp.fuzzy_name_score(row[idx_name], b) for b in base_names), default=0)
				score = max(score, best / 100.0)
			
			if score < threshold:
				continue
			
			item = dict(zip(cols, row))
			item.pop("pan_upper", None)
			item.pop("pan_raw", None)
			if idx_year >= 0 and idx_year < len(row):
				item["year"] = row[idx_year]
			item["match_score"] = round(float(score), 3)
			data.append(item)
		
		# If PAN provided, fetch metadata
		if pan:
			try:
				ext = urllib.request.urlopen(f"http://192.168.1.198:5000/get_by_id?" + urllib.parse.urlencode({"id": pan_up}), timeout=5)
				ej = json.loads(ext.read().decode("utf-8"))
				meta_name = ej.get("name") or ej.get("Name") or ej.get("full_name") or ej.get("fullName")
				meta_age = ej.get("age") or ej.get("Age") or ej.get("person_age")
				if meta_name:
					for r in data:
						r["Name"] = str(meta_name) if meta_name else ""
				if meta_age is not None:
					try:
						age_num = int(str(meta_age))
						if data and data[0].get("year"):
							year_num = int(str(data[0]["year"]))
							if 1900 < year_num <= datetime.now().year + 1:
								current_age = age_num + (datetime.now().year - year_num)
								for r in data:
									r["Age"] = current_age
					except Exception:
						pass
			except Exception:
				pass
		
		# Process data to add Property Type column
		# Get the entity name for comparison (from Name column or search_pan)
		entity_name_for_comparison = None
		if pan:
			entity_name_for_comparison = pan_up
		elif seed_name:
			entity_name_for_comparison = nlp.normalize_name(seed_name or "")
		
		# If we have data, try to get the Name from first row
		if data and not entity_name_for_comparison:
			entity_name_for_comparison = data[0].get("Name") or data[0].get("name_extracted") or ""
		
		# Process each row to determine Property Type
		for row in data:
			property_type = "N/A"
			
			# Get the Name from this row (the entity we're searching for)
			row_name = row.get("Name") or row.get("name_extracted") or ""
			if not row_name and entity_name_for_comparison:
				row_name = entity_name_for_comparison
			
			# Extract names and PANs from Buyer and Seller columns
			buyer_text = str(row.get("buyer", "") or "")
			seller_text = str(row.get("seller", "") or "")
			
			buyer_names = _extract_names_from_marathi_string(buyer_text)
			seller_names = _extract_names_from_marathi_string(seller_text)
			buyer_pans = _extract_pan_numbers_from_text(buyer_text)
			seller_pans = _extract_pan_numbers_from_text(seller_text)
			
			# Check if name appears in buyer or seller
			is_buyer = False
			is_seller = False
			
			if row_name:
				# Normalize for comparison
				row_name_normalized = nlp.normalize_name(row_name) if hasattr(nlp, 'normalize_name') else row_name.upper()
				row_name_upper = row_name.upper()
				
				# Check buyer names
				for b_name in buyer_names:
					b_name_normalized = nlp.normalize_name(b_name) if hasattr(nlp, 'normalize_name') else b_name.upper()
					b_name_upper = b_name.upper()
					# Check for partial matches (name in buyer name or vice versa)
					if (row_name_normalized in b_name_normalized or b_name_normalized in row_name_normalized or
						row_name_upper in b_name_upper or b_name_upper in row_name_upper):
						is_buyer = True
						break
				
				# Check seller names
				for s_name in seller_names:
					s_name_normalized = nlp.normalize_name(s_name) if hasattr(nlp, 'normalize_name') else s_name.upper()
					s_name_upper = s_name.upper()
					# Check for partial matches (name in seller name or vice versa)
					if (row_name_normalized in s_name_normalized or s_name_normalized in row_name_normalized or
						row_name_upper in s_name_upper or s_name_upper in row_name_upper):
						is_seller = True
						break
			
			# Also check PAN numbers if searching by PAN
			if pan and pan_up:
				pan_up_normalized = pan_up.upper()
				# Check if search PAN appears in buyer PANs
				if pan_up_normalized in buyer_pans:
					is_buyer = True
				# Check if search PAN appears in seller PANs
				if pan_up_normalized in seller_pans:
					is_seller = True
			
			# Determine property type
			if is_buyer and not is_seller:
				property_type = "Bought"
			elif is_seller and not is_buyer:
				property_type = "Sold"
			elif is_buyer and is_seller:
				property_type = "Both"
			
			row["property_type"] = property_type
		
		# Define columns for PDF
		desired = [
			{"header": "PAN", "key": "pan_numbers"},
			{"header": "Buyer", "key": "buyer"},
			{"header": "Seller", "key": "seller"},
			{"header": "Name", "key": "Name"},
			{"header": "Current Age", "key": "Age"},
			{"header": "Property Type", "key": "property_type"},
		]
		cols_display = [d["header"] for d in desired]
		col_map = {d["header"]: d["key"] for d in desired}
		
		if not data:
			raise HTTPException(status_code=400, detail="No data found matching the search criteria and threshold")
		
		# Generate PDF - pass the searched PAN if available
		try:
			searched_pan = pan_up if pan else None
			pdf_buffer = _generate_pdf(data, cols_display, col_map, search_pan=searched_pan)
		except Exception as e:
			logger.exception("PDF generation failed: %s", e)
			raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
		
		filename = f"transaction_report_{datetime.now().strftime('%Y-%m-%d')}.pdf"
		return Response(
			content=pdf_buffer.read(),
			media_type="application/pdf",
			headers={
				"Content-Disposition": f'attachment; filename="{filename}"',
				"Access-Control-Allow-Origin": "*",
				"Access-Control-Allow-Methods": "GET",
				"Access-Control-Allow-Headers": "*",
			}
		)
	except HTTPException:
		raise
	except Exception as e:
		logger.exception("Unexpected error in export_pdf: %s", e)
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
	finally:
		con.close()
