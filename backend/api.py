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
from fastapi import FastAPI, HTTPException, Query, Body
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
    from .config import THRESHOLDS as TH, SEARCH as SP
    from .utils import nlp
    from .search import (
        search_by_pan as _search_by_pan,
        search_by_seed_name as _search_by_seed_name,
        fetch_rows_for_pan as _fetch_rows_for_pan,
        pick_canonical_names as _pick_canonical_names,
    )
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pansearch.api")


_DEVANAGARI_FONTS = None
_LLM_NAME_CACHE: dict[str, list[str]] = {}

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
	return bool(re.search(r"[\u0900-\u097F]", text or ""))


def clean_english_name(raw_english: str) -> str:
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
		cols = [r[1] for r in con.execute("PRAGMA table_info('transactions')").fetchall()]
		idx_name = cols.index("name_norm") if "name_norm" in cols else -1
		idx_pan = cols.index("pan_upper") if "pan_upper" in cols else -1
		idx_year = cols.index("year") if "year" in cols else -1
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
			item.pop("pan_upper", None)
			item.pop("pan_raw", None)
			if idx_year >= 0:
				if idx_year < len(row):
					item["year"] = row[idx_year]
				else:
					logger.warning("Row length mismatch: expected year at index %d but row has %d elements", idx_year, len(row))
					item["year"] = None
			item["match_score"] = round(float(score), 3)
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
					if all(t in low for t in tokens):
						marked = val
						for t in sorted(tokens, key=len, reverse=True):
							try:
								pattern = re.compile(re.escape(t), re.IGNORECASE)
								marked = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", marked)
							except Exception:
								pass
						hit_field = f
						hit_snippet_html = marked
						break
				if hit_field:
					item["name_hit_field"] = hit_field
					item["name_hit_snippet"] = hit_snippet
					item["name_hit_snippet_html"] = hit_snippet_html

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
		entity_name_for_comparison = None
		if pan:
			entity_name_for_comparison = pan_up
		elif seed_name:
			entity_name_for_comparison = nlp.normalize_name(seed_name or "")
		for item in data:
			property_type = "N/A"
			row_name = item.get("Name") or item.get("name_extracted") or ""
			if not row_name and entity_name_for_comparison:
				row_name = entity_name_for_comparison
			buyer_text = str(item.get("buyer", "") or "")
			seller_text = str(item.get("seller", "") or "")
			buyer_names = _extract_names_from_marathi_string(buyer_text)
			seller_names = _extract_names_from_marathi_string(seller_text)
			buyer_pans = _extract_pan_numbers_from_text(buyer_text)
			seller_pans = _extract_pan_numbers_from_text(seller_text)
			is_buyer = False
			is_seller = False
			if row_name:
				row_name_normalized = nlp.normalize_name(row_name) if hasattr(nlp, 'normalize_name') else row_name.upper()
				row_name_upper = row_name.upper()
				for b_name in buyer_names:
					b_name_normalized = nlp.normalize_name(b_name) if hasattr(nlp, 'normalize_name') else b_name.upper()
					b_name_upper = b_name.upper()
					if (row_name_normalized in b_name_normalized or b_name_normalized in row_name_normalized or
						row_name_upper in b_name_upper or b_name_upper in row_name_upper):
						is_buyer = True
						break
				for s_name in seller_names:
					s_name_normalized = nlp.normalize_name(s_name) if hasattr(nlp, 'normalize_name') else s_name.upper()
					s_name_upper = s_name.upper()
					if (row_name_normalized in s_name_normalized or s_name_normalized in row_name_normalized or
						row_name_upper in s_name_upper or s_name_upper in row_name_upper):
						is_seller = True
						break
			if pan and pan_up:
				pan_up_normalized = pan_up.upper()
				if pan_up_normalized in buyer_pans:
					is_buyer = True
				if pan_up_normalized in seller_pans:
					is_seller = True
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


def _extract_person_names_llm(raw_text: str) -> list[str]:
	if not raw_text:
		return []
	try:
		http_client = httpx.Client(timeout=30.0)
		client = OpenAI(
			base_url="http://192.168.1.198:11434/v1",
			api_key="ollama",
			http_client=http_client
		)
		system_prompt = "Return JSON only. You extract concise person names."
		user_prompt = (
			"You are an intelligent name extraction assistant.\n"
			"Your task is to extract **all human names** mentioned in the given text input.\n"
			"The input text may contain Marathi, Hindi, or English words — or a mix of them.\n\n"
			"### Follow these rules strictly:\n"
			"1. Extract only human names.\n"
			"   - Include *all* person names, whether individuals or representatives of organizations.\n"
			"   - Exclude company names, organization names, places, or designations unless they are part of a person’s full name.\n"
			"2. Each extracted name must be complete.\n"
			"   - Preserve first name, middle name, and last name if available.\n"
			"   - Do not split parts of one name into multiple names.\n"
			"3. Preserve the script — Marathi names should remain in Marathi, English names in English.\n"
			"4. Respond ONLY as compact JSON array. Example:\n"
			'   [{"name": "यश रितेश मुठा"}, {"name": "अजित भोईर"}]\n'
			"5. If no human name is found, return an empty array: []\n\n"
			"Text may include roles (e.g., संचालक, तर्फे), organizations, or punctuation.\n\n"
			f"Text: {raw_text}"
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

		names: list[str] = []
		try:
			obj = json.loads(text)
			if isinstance(obj, list):
				for entry in obj:
					if isinstance(entry, str) and entry.strip():
						names.append(entry.strip())
					elif isinstance(entry, dict):
						val = entry.get("name")
						if isinstance(val, str) and val.strip():
							names.append(val.strip())
			elif isinstance(obj, dict):
				val = obj.get("name")
				if isinstance(val, str) and val.strip():
					names.append(val.strip())
		except Exception:
			matches = re.findall(r"\{\s*\"name\"\s*:\s*\"(.+?)\"\s*\}", text)
			for match in matches:
				val = match.strip().strip('"\'\u201c\u201d')
				if val:
					names.append(val)
			if not names and text.startswith("[") and text.endswith("]"):
				try:
					raw_list = json.loads(text.replace("'", '"'))
					if isinstance(raw_list, list):
						for item in raw_list:
							if isinstance(item, str) and item.strip():
								names.append(item.strip())
				except Exception:
					pass

		names = [n.strip().strip('"\'\u201c\u201d') for n in names if isinstance(n, str) and n.strip()]
		names = [n for n in names if 0 < len(n) <= 80]
		if names:
			seen = set()
			unique: list[str] = []
			for n in names:
				if n not in seen:
					seen.add(n)
					unique.append(n)
			return unique

		stop = {"चे", "चा", "ची", "चे", "प्रा", "ली", "लि", "अँड", "एंड", "तर्फे", "संचालक", "मुखत्यार", "प्रा.", "लि.", "कंपनी", "लिमिटेड", "प्रा.लि."}
		words = re.split(r"\s+", raw_text.strip())
		filtered = [w for w in words if w and (re.sub(r"[\,;:\-]", "", w) not in stop)]
		if filtered:
			cand = " ".join(filtered[-2:]) if len(filtered) >= 2 else filtered[-1]
			cand = cand.strip()
			if 0 < len(cand) <= 80:
				logger.info("LLM fallback heuristic used for extraction")
				return [cand]
		return []
	except Exception as e:
		logger.exception("LLM extraction failed: %s", e)
		return []


def _get_llm_names_cached(text: str) -> list[str]:
	key = (text or "").strip()
	if not key:
		return []
	if key in _LLM_NAME_CACHE:
		return _LLM_NAME_CACHE[key]
	names = _extract_person_names_llm(key)
	_LLM_NAME_CACHE[key] = names
	return names


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
		name = data.get("name") or data.get("Name") or data.get("full_name") or data.get("fullName")
		age = data.get("age") if "age" in data else data.get("Age") if "Age" in data else data.get("person_age")
		if name is not None and not isinstance(name, str):
			name = str(name)
		if age is not None and not isinstance(age, (int, float)):
			try:
				age_num = int(str(age))
				age = age_num
			except Exception:
				pass
		names_llm = _get_llm_names_cached(name or "") if name else []
		extracted = None
		final_name = name
		extraction_error = None
		return {
			"pan": pan_id,
			"name": final_name,
			"extracted_name": extracted,
			"names": names_llm,
			"raw_name": name,
			"age": age,
			"raw": data,
			"extraction_error": extraction_error,
		}
	except Exception as e:
		logger.exception("PAN meta upstream error for %s: %s", pan_id, e)
		raise HTTPException(status_code=502, detail=f"upstream error: {e}")


@app.get("/llm_test")
def llm_test(text: str = Query(..., description="Sample text to extract name from")) -> dict:
	try:
		names = _extract_person_names_llm(text)
		return {"input": text, "names": names}
	except Exception as e:
		logger.exception("LLM test failed: %s", e)
		return {"input": text, "names": []}




def _register_devanagari_fonts():
	
	global _DEVANAGARI_FONTS
	if _DEVANAGARI_FONTS is not None:
		return _DEVANAGARI_FONTS
	font_paths = [
		"/System/Library/Fonts/Supplemental/NotoSansDevanagari-Regular.ttf",
		"/Library/Fonts/NotoSansDevanagari-Regular.ttf",
		"~/Library/Fonts/NotoSansDevanagari-Regular.ttf",
		"/System/Library/Fonts/Supplemental/NotoSansDevanagari-Bold.ttf",
		"/Library/Fonts/NotoSansDevanagari-Bold.ttf",
		"~/Library/Fonts/NotoSansDevanagari-Bold.ttf",
		"/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
		"/System/Library/Fonts/Supplemental/ITFDevanagari.ttc",
		"/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
		"/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
		"/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
		"/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
		"/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf",
		"C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
		"C:/Windows/Fonts/NotoSansDevanagari-Bold.ttf",
		"C:/Windows/Fonts/mangal.ttf",
		"C:/Windows/Fonts/MANGAL.TTF",
		"C:/Windows/Fonts/mangalb.ttf",
		"C:/Windows/Fonts/MANGALB.TTF",
		"C:/Windows/Fonts/nirmala.ttf",
		"C:/Windows/Fonts/NIRMALA.TTF",
		"C:/Windows/Fonts/nirmalab.ttf",
		"C:/Windows/Fonts/NIRMALAB.TTF",
		"C:/Windows/Fonts/nirmalas.ttf",
		"C:/Windows/Fonts/NIRMALAS.TTF",
		"/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
		"/Library/Fonts/Arial Unicode.ttf",
		"C:/Windows/Fonts/ARIALUNI.TTF",
	]
	expanded_paths = []
	for path in font_paths:
		if path.startswith("~"):
			expanded_paths.append(os.path.expanduser(path))
		else:
			expanded_paths.append(path)
	font_paths = expanded_paths
	if os.name == 'nt':
		try:
			import glob
			windows_font_dirs = [
				os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts'),
				'C:/Windows/Fonts',
				'D:/Windows/Fonts',
			]
			for font_dir in windows_font_dirs:
				if os.path.exists(font_dir):
					patterns = [
						'*devanagari*',
						'*Devanagari*',
						'*mangal*',
						'*Mangal*',
						'*MANGAL*',
						'*nirmala*',
						'*Nirmala*',
						'*NIRMALA*',
						'*noto*devanagari*',
						'*Noto*Devanagari*',
					]
					for pattern in patterns:
						for font_file in glob.glob(os.path.join(font_dir, pattern)):
							if font_file not in font_paths and (font_file.lower().endswith('.ttf') or font_file.lower().endswith('.ttc')):
								font_paths.append(font_file)
		except Exception as e:
			logger.debug(f"Could not search Windows fonts directory: {e}")
	try:
		import glob
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
	regular_font_path = None
	for font_path in font_paths:
		if os.path.exists(font_path):
			try:
				font_lower = font_path.lower()
				is_bold = (
					"bold" in font_lower or
					"b" in font_lower.split('.')[0].split('\\')[-1].split('/')[-1][-1:] or
					"ITF" in font_path or
					"semibold" in font_lower or
					"semi" in font_lower
				)
				if is_bold and not registered["bold"]:
					pdfmetrics.registerFont(TTFont("Devanagari-Bold", font_path))
					registered["bold"] = "Devanagari-Bold"
					logger.info(f"Registered Devanagari bold font: {font_path}")
				elif not is_bold and not registered["regular"]:
					pdfmetrics.registerFont(TTFont("Devanagari", font_path))
					registered["regular"] = "Devanagari"
					regular_font_path = font_path
					logger.info(f"Registered Devanagari regular font: {font_path}")
				if registered["regular"] and registered["bold"]:
					break
			except Exception as e:
				logger.warning(f"Failed to register font {font_path}: {e}")
				continue
	if registered["regular"] and not registered["bold"] and regular_font_path:
		try:
			pdfmetrics.registerFont(TTFont("Devanagari-Bold", regular_font_path))
			registered["bold"] = "Devanagari-Bold"
			logger.info(f"Using regular font for bold (no bold variant found): {regular_font_path}")
		except Exception as e:
			logger.warning(f"Failed to register regular font as bold: {e}")
			registered["bold"] = registered["regular"]
	if not registered["regular"]:
		logger.warning("No Devanagari font found. Devanagari text may not render correctly.")
		logger.warning("To enable Devanagari support, install a Devanagari font:")
		logger.warning("  macOS: Font is usually pre-installed")
		logger.warning("  Linux: sudo apt-get install fonts-noto-core")
		logger.warning("  Windows: Mangal or Nirmala UI fonts are usually pre-installed, or download Noto Sans Devanagari from https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari")
		registered["regular"] = "Helvetica"
		registered["bold"] = "Helvetica-Bold"
	else:
		logger.info(f"Successfully registered Devanagari fonts: regular={registered['regular']}, bold={registered['bold']}")
	_DEVANAGARI_FONTS = registered
	return registered


def _strip_html(text):
	if not text:
		return ""
	return re.sub(r"<[^>]+>", "", str(text))


def _extract_names_from_marathi_string(text: str) -> List[str]:
	if not text:
		return []
	names = []
	text_str = str(text)
	marathi_pattern = r'नाव[ः-]-\s*([^;]+?)(?=\s*वयः-|\s*पत्ता:-|\s*पॅन नं:-|\s*\d\):|\s*नाव[ः-]|$)'
	marathi_matches = re.findall(marathi_pattern, text_str, re.IGNORECASE)
	for match in marathi_matches:
		name = match.strip()
		if name:
			names.append(name)
	english_pattern = r'(?:^|\d\)\s*)Name:\s*\.?\s*([^;]+?)(?=\s*Age:|\s*Address:|\s*PAN:|\s*\d\):|$)'
	english_matches = re.findall(english_pattern, text_str, re.IGNORECASE)
	for match in english_matches:
		name = match.strip()
		if name and name not in names:
			names.append(name)
	return names


def _extract_pan_numbers_from_text(text: str) -> List[str]:
	if not text:
		return []
	pans = []
	text_str = str(text)
	pan_pattern = r'(?:पॅन नं[ः-]|PAN[:\s]+)([A-Z]{5}\d{4}[A-Z])'
	pan_matches = re.findall(pan_pattern, text_str, re.IGNORECASE)
	for match in pan_matches:
		pan = match.strip().upper()
		if pan and pan not in pans:
			pans.append(pan)
	standalone_pan_pattern = r'\b([A-Z]{5}\d{4}[A-Z])\b'
	standalone_matches = re.findall(standalone_pan_pattern, text_str)
	for match in standalone_matches:
		pan = match.strip().upper()
		if pan and pan not in pans:
			pans.append(pan)
	return pans


def _generate_pdf(data_rows: List[dict], cols_display: List[str], col_map: dict, search_pan: Optional[str] = None) -> BytesIO:
	if not data_rows:
		raise ValueError("No data rows provided for PDF generation")
	buffer = BytesIO()
	page_size = landscape(A4)
	doc = SimpleDocTemplate(buffer, pagesize=page_size, topMargin=30*mm, bottomMargin=30*mm, leftMargin=15*mm, rightMargin=15*mm)
	fonts = _register_devanagari_fonts()
	devanagari_font = fonts["regular"]
	devanagari_bold = fonts["bold"]
	standard_fonts = {"Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
	                  "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
	                  "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"}
	if devanagari_font not in standard_fonts:
		try:
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
	styles = getSampleStyleSheet()
	primary_color = colors.HexColor("#1E3A5F")
	primary_dark = colors.HexColor("#152238")
	accent_color = colors.HexColor("#D4AF37")
	text_dark = colors.HexColor("#1A1A1A")
	text_medium = colors.HexColor("#424242")
	text_light = colors.HexColor("#6B7280")
	bg_light = colors.HexColor("#F8F9FA")
	bg_blue_light = colors.HexColor("#E8F0F8")
	border_color = colors.HexColor("#D1D5DB")
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
	story = []
	def para(text, style=None):
		if style is None:
			style = normal_style
		if not text:
			return Paragraph("", style)
		text = _strip_html(str(text))
		text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
		return Paragraph(text, style)
	story.append(Spacer(1, 8*mm))
	title_para = Paragraph("Property Summary Report for Financial Creditor", title_style)
	story.append(title_para)
	story.append(Spacer(1, 4*mm))
	if search_pan:
		entity_display = nlp.canonicalize_pan(search_pan) or search_pan.upper()
	else:
		entity_display = data_rows[0].get("name_extracted") or data_rows[0].get("Name") or data_rows[0].get("pan_numbers") or "N/A"
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
		("LINEBELOW", (0, 0), (-1, -1), 2, accent_color),
	]))
	story.append(entity_table)
	story.append(Spacer(1, 10*mm))
	to_style = ParagraphStyle("ToSection", parent=styles["Normal"], fontSize=9, fontName=devanagari_font,
		textColor=text_medium, leftIndent=0, spaceAfter=2)
	try:
		to_heading_style = ParagraphStyle("ToHeading", parent=styles["Normal"], fontSize=10, fontName=devanagari_bold,
			textColor=primary_dark, leftIndent=0, spaceAfter=4)
	except Exception:
		to_heading_style = ParagraphStyle("ToHeading", parent=styles["Normal"], fontSize=10, fontName="Helvetica-Bold",
			textColor=primary_dark, leftIndent=0, spaceAfter=4)
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
		("LINEBELOW", (0, 0), (-1, 0), 2, primary_color),
	]))
	story.append(to_table)
	story.append(Spacer(1, 10*mm))
	table_title_para = Paragraph("1) Details : Transaction Summary", heading_style)
	story.append(table_title_para)
	story.append(Spacer(1, 6*mm))
	standard_fonts_set = {"Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
	                      "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
	                      "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"}
	if devanagari_font not in standard_fonts_set:
		table_font = devanagari_font
		logger.info(f"Using Devanagari font for tables: {table_font}")
	else:
		table_font = "Helvetica"
		logger.warning("Devanagari font not available, using Helvetica (Marathi text may not render correctly)")
	if devanagari_bold not in standard_fonts_set:
		table_bold = devanagari_bold
		logger.info(f"Using Devanagari bold font for tables: {table_bold}")
	else:
		table_bold = "Helvetica-Bold"
	summary_headers = ["Sr"] + cols_display
	summary_data = []
	def cell_para(text, is_header=False):
		if text is None:
			text = ""
		text = _strip_html(str(text))
		text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
		if isinstance(text, bytes):
			text = text.decode('utf-8', errors='ignore')
		try:
			if is_header:
				style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontSize=9, fontName=table_bold, textColor=colors.white)
			else:
				style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, fontName=table_font, textColor=text_dark)
		except Exception:
			if is_header:
				style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=colors.white)
			else:
				style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, fontName="Helvetica", textColor=text_dark)
		return Paragraph(text, style)
	header_paras = [cell_para(h, is_header=True) for h in summary_headers]
	for idx, row in enumerate(data_rows):
		row_data = [cell_para(str(idx + 1))]
		for header in cols_display:
			key = col_map.get(header, header)
			val = row.get(key, "")
			if val is not None:
				val_str = str(val)
				if len(val_str) > 500:
					val_str = val_str[:500] + "..."
				row_data.append(cell_para(val_str))
			else:
				row_data.append(cell_para(""))
		summary_data.append(row_data)
	available_width = page_size[0] - 30*mm
	serial_width = 20*mm
	remaining_width = available_width - serial_width
	col_widths = [serial_width] + [remaining_width / len(cols_display)] * len(cols_display)
	table = Table([header_paras] + summary_data, colWidths=col_widths)
	table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, 0), primary_dark),
		("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
		("ALIGN", (0, 0), (-1, 0), "LEFT"),
		("FONTNAME", (0, 0), (-1, 0), table_bold),
		("FONTSIZE", (0, 0), (-1, 0), 9),
		("BOTTOMPADDING", (0, 0), (-1, 0), 8),
		("TOPPADDING", (0, 0), (-1, 0), 8),
		("LEFTPADDING", (0, 0), (-1, 0), 6),
		("RIGHTPADDING", (0, 0), (-1, 0), 6),
		("LINEBELOW", (0, 0), (-1, 0), 2, accent_color),
		("BACKGROUND", (0, 1), (-1, -1), colors.white),
		("TEXTCOLOR", (0, 1), (-1, -1), text_dark),
		("ALIGN", (0, 1), (0, -1), "CENTER"),
		("ALIGN", (1, 1), (-1, -1), "LEFT"),
		("FONTNAME", (0, 1), (-1, -1), table_font),
		("FONTSIZE", (0, 1), (-1, -1), 8),
		("GRID", (0, 0), (-1, -1), 0.5, border_color),
		("VALIGN", (0, 0), (-1, -1), "TOP"),
		("LEFTPADDING", (0, 1), (-1, -1), 6),
		("RIGHTPADDING", (0, 1), (-1, -1), 6),
		("TOPPADDING", (0, 1), (-1, -1), 5),
		("BOTTOMPADDING", (0, 1), (-1, -1), 5),
		("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
		("BACKGROUND", (0, 1), (0, -1), bg_blue_light),
	]))
	story.append(table)
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
	
	canvas.saveState()
	page_width = landscape(A4)[0]
	page_height = landscape(A4)[1]
	primary_color = colors.HexColor("#1E3A5F")
	accent_color = colors.HexColor("#D4AF37")
	top_bar_y = page_height - 30*mm
	canvas.setFillColor(primary_color)
	canvas.rect(0, top_bar_y, page_width, 4*mm, fill=1, stroke=0)
	canvas.setFillColor(accent_color)
	canvas.rect(0, top_bar_y, page_width, 1.2*mm, fill=1, stroke=0)
	if logo_path and os.path.exists(logo_path):
		try:
			logo_width = 60*mm
			logo_height = 22*mm
			logo_x = page_width - 15*mm - logo_width
			logo_y = page_height - 15*mm - logo_height
			canvas.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True)
		except Exception as e:
			logger.warning(f"Failed to add logo: {e}")
	if footer_path and os.path.exists(footer_path):
		try:
			canvas.setFillColor(primary_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 2*mm, fill=1, stroke=0)
			canvas.setFillColor(accent_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 0.5*mm, fill=1, stroke=0)
			footer_width = page_width - 30*mm
			footer_height = 20*mm
			footer_x = 15*mm
			footer_y = 15*mm
			canvas.drawImage(footer_path, footer_x, footer_y, width=footer_width, height=footer_height, preserveAspectRatio=True)
		except Exception as e:
			logger.warning(f"Failed to add footer image: {e}")
			canvas.setFont("Helvetica", 8)
			canvas.setFillColor(colors.HexColor("#424242"))
			canvas.drawString(15*mm, 20*mm, "B1-001, U/B Floor, Boomerang Chandivali Farm Road, Powai, Andheri East, Mumbai, Maharashtra 400072")
			canvas.setFillColor(colors.HexColor("#1E3A5F"))
			canvas.drawRightString(page_width - 15*mm, 20*mm, "www.real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, 16*mm, "info@real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, 12*mm, "+91 88284 05969")
	canvas.restoreState()


def _add_page_header_footer(canvas, doc, logo_path=None, footer_path=None):
	
	canvas.saveState()
	page_width = landscape(A4)[0]
	page_height = landscape(A4)[1]
	primary_color = colors.HexColor("#1E3A5F")
	accent_color = colors.HexColor("#D4AF37")
	top_bar_y = page_height - 30*mm
	canvas.setFillColor(primary_color)
	canvas.rect(0, top_bar_y, page_width, 4*mm, fill=1, stroke=0)
	canvas.setFillColor(accent_color)
	canvas.rect(0, top_bar_y, page_width, 1.2*mm, fill=1, stroke=0)
	if logo_path and os.path.exists(logo_path):
		try:
			logo_width = 60*mm
			logo_height = 22*mm
			logo_x = page_width - 15*mm - logo_width
			logo_y = page_height - 15*mm - logo_height
			canvas.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True)
		except Exception as e:
			logger.warning(f"Failed to add logo: {e}")
	if footer_path and os.path.exists(footer_path):
		try:
			canvas.setFillColor(primary_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 2*mm, fill=1, stroke=0)
			canvas.setFillColor(accent_color)
			canvas.rect(15*mm, 35*mm, page_width - 30*mm, 0.5*mm, fill=1, stroke=0)
			footer_width = page_width - 30*mm
			footer_height = 20*mm
			footer_x = 15*mm
			footer_y = 15*mm
			canvas.drawImage(footer_path, footer_x, footer_y, width=footer_width, height=footer_height, preserveAspectRatio=True)
			page_num = canvas.getPageNumber()
			canvas.setFont("Helvetica-Bold", 10)
			canvas.setFillColor(primary_color)
			canvas.drawCentredString(page_width / 2, footer_y + footer_height + 3*mm, f"Page {page_num}")
		except Exception as e:
			logger.warning(f"Failed to add footer image: {e}")
			footer_y = 25*mm
			canvas.setFont("Helvetica-Bold", 10)
			canvas.setFillColor(primary_color)
			page_num = canvas.getPageNumber()
			canvas.drawCentredString(page_width / 2, footer_y - 5*mm, f"Page {page_num}")
			canvas.setFont("Helvetica", 8)
			canvas.setFillColor(colors.HexColor("#424242"))
			canvas.drawString(15*mm, footer_y - 5*mm, "B1-001, U/B Floor, Boomerang Chandivali Farm Road, Powai, Andheri East, Mumbai, Maharashtra 400072")
			canvas.setFillColor(colors.HexColor("#1E3A5F"))
			canvas.drawRightString(page_width - 15*mm, footer_y - 5*mm, "www.real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, footer_y - 9*mm, "info@real-value.co.in")
			canvas.drawRightString(page_width - 15*mm, footer_y - 13*mm, "+91 88284 05969")
	canvas.restoreState()


@app.post("/export_pdf")
def export_pdf(
	payload: dict = Body(default=None),
	pan: Optional[str] = Query(default=None),
	seed_name: Optional[str] = Query(default=None),
	limit: Optional[int] = Query(default=1000),
	threshold: Optional[float] = Query(default=0.75),
) -> Response:
	
	if not payload and not pan and not seed_name:
		raise HTTPException(status_code=400, detail="Provide cart payload or query parameters")
	if not os.path.exists(DB_PATH):
		raise HTTPException(status_code=500, detail=f"DB not found at {DB_PATH}")
	con = duckdb.connect(DB_PATH, read_only=True)
	try:
		if payload:
			rows = payload.get("rows") or []
			cols = list({key for row in rows for key in row.keys()})
		elif pan or seed_name:
			if pan:
				rows = _search_by_pan(con, pan)
			else:
				rows = _search_by_seed_name(con, seed_name or "")
			rows = rows[: (limit or len(rows))]
			cols = [r[1] for r in con.execute("PRAGMA table_info('transactions')").fetchall()]
		else:
			raise HTTPException(status_code=400, detail="Provide cart payload or query parameters")
		idx_name = cols.index("name_norm") if "name_norm" in cols else -1
		idx_pan = cols.index("pan_upper") if "pan_upper" in cols else -1
		idx_year = cols.index("year") if "year" in cols else -1
		base_names: list[str] = []
		pan_up = None
		data: list[dict] = []

		if payload and rows:
			for row in rows:
				item = dict(row)
				item["match_score"] = round(float(item.get("match_score", 1.0)), 3)
				data.append(item)
		else:
			if pan:
				try:
					base_rows = _fetch_rows_for_pan(con, pan)
					base_names = _pick_canonical_names([(r[0],) for r in base_rows])
				except Exception:
					base_names = []
			elif seed_name:
				base_names = [nlp.normalize_name(seed_name or "")]
			pan_up = nlp.canonicalize_pan(pan) if pan else None
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
		entity_name_for_comparison = None
		if pan:
			entity_name_for_comparison = pan_up
		elif seed_name:
			entity_name_for_comparison = nlp.normalize_name(seed_name or "")
		if data and not entity_name_for_comparison:
			entity_name_for_comparison = data[0].get("Name") or data[0].get("name_extracted") or ""
		for row in data:
			property_type = "N/A"
			row_name = row.get("Name") or row.get("name_extracted") or ""
			if not row_name and entity_name_for_comparison:
				row_name = entity_name_for_comparison
			buyer_text = str(row.get("buyer", "") or "")
			seller_text = str(row.get("seller", "") or "")
			buyer_names = _extract_names_from_marathi_string(buyer_text)
			seller_names = _extract_names_from_marathi_string(seller_text)
			buyer_pans = _extract_pan_numbers_from_text(buyer_text)
			seller_pans = _extract_pan_numbers_from_text(seller_text)
			is_buyer = False
			is_seller = False
			if row_name:
				row_name_normalized = nlp.normalize_name(row_name) if hasattr(nlp, 'normalize_name') else row_name.upper()
				row_name_upper = row_name.upper()
				for b_name in buyer_names:
					b_name_normalized = nlp.normalize_name(b_name) if hasattr(nlp, 'normalize_name') else b_name.upper()
					b_name_upper = b_name.upper()
					if (row_name_normalized in b_name_normalized or b_name_normalized in row_name_normalized or
						row_name_upper in b_name_upper or b_name_upper in row_name_upper):
						is_buyer = True
						break
				for s_name in seller_names:
					s_name_normalized = nlp.normalize_name(s_name) if hasattr(nlp, 'normalize_name') else s_name.upper()
					s_name_upper = s_name.upper()
					if (row_name_normalized in s_name_normalized or s_name_normalized in row_name_normalized or
						row_name_upper in s_name_upper or s_name_upper in row_name_upper):
						is_seller = True
						break
			if pan and pan_up:
				pan_up_normalized = pan_up.upper()
				if pan_up_normalized in buyer_pans:
					is_buyer = True
				if pan_up_normalized in seller_pans:
					is_seller = True
			if is_buyer and not is_seller:
				property_type = "Bought"
			elif is_seller and not is_buyer:
				property_type = "Sold"
			elif is_buyer and is_seller:
				property_type = "Both"
			row["property_type"] = property_type
			name_sources: list[str] = []
			for field in ("Name", "name_extracted", "_name_primary", "_name_alt", "buyer", "seller"):
				val = row.get(field)
				if isinstance(val, str) and val.strip():
					name_sources.append(val.strip())
			if name_sources:
				combined_text = " | ".join(name_sources)
				names_llm = _get_llm_names_cached(combined_text)
				if names_llm:
					row["names_llm"] = names_llm
					row["Name"] = ", ".join(names_llm)
		if payload and rows:
			custom_cols = payload.get("cols")
			if custom_cols and isinstance(custom_cols, list):
				cols_display = custom_cols
				col_map = {c: c for c in cols_display}
			else:
				default = [
					{"header": "PAN", "key": "pan_numbers"},
					{"header": "Buyer", "key": "buyer"},
					{"header": "Seller", "key": "seller"},
					{"header": "Name", "key": "Name"},
					{"header": "Current Age", "key": "Age"},
					{"header": "Property Type", "key": "property_type"},
				]
				cols_display = [d["header"] for d in default]
				col_map = {d["header"]: d["key"] for d in default}
		else:
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
		try:
			searched_pan = pan_up if pan else payload.get("search_pan") if payload else None
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
