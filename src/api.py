import os
import logging
from typing import List, Optional
import re
import json
import time
import sqlite3
import urllib.parse
import urllib.request
from openai import OpenAI

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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


# -------------------- Phonetics Cache (24h, memory + disk) --------------------
_PHON_CACHE_TTL_SEC = 24 * 3600
_PHON_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "phonetics_cache.sqlite")
_phon_cache_mem: dict[str, tuple[int, list[str], list[str]]] = {}

def _ensure_cache_db() -> None:
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"), exist_ok=True)
    con = sqlite3.connect(_PHON_CACHE_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS variants (
                k TEXT PRIMARY KEY,
                ts INTEGER NOT NULL,
                english TEXT NOT NULL,
                marathi TEXT NOT NULL
            )
            """
        )
        con.commit()
    finally:
        con.close()

def _cache_get_variants(key: str) -> Optional[dict]:
    if not key:
        return None
    now = int(time.time())
    # memory
    tup = _phon_cache_mem.get(key)
    if tup and now - tup[0] <= _PHON_CACHE_TTL_SEC:
        return {"english_variants": tup[1], "marathi_variants": tup[2]}
    # disk
    try:
        _ensure_cache_db()
        con = sqlite3.connect(_PHON_CACHE_PATH)
        try:
            row = con.execute("SELECT ts, english, marathi FROM variants WHERE k = ?", [key]).fetchone()
            if not row:
                return None
            ts_val, eng_json, mar_json = row
            if now - int(ts_val) > _PHON_CACHE_TTL_SEC:
                return None
            eng = json.loads(eng_json)
            mar = json.loads(mar_json)
            # populate memory
            _phon_cache_mem[key] = (int(ts_val), eng, mar)
            return {"english_variants": eng, "marathi_variants": mar}
        finally:
            con.close()
    except Exception:
        return None

def _cache_put_variants(key: str, english: list[str], marathi: list[str]) -> None:
    if not key:
        return
    now = int(time.time())
    # memory
    _phon_cache_mem[key] = (now, english, marathi)
    # disk
    try:
        _ensure_cache_db()
        con = sqlite3.connect(_PHON_CACHE_PATH)
        try:
            con.execute(
                "INSERT INTO variants(k, ts, english, marathi) VALUES(?, ?, ?, ?)\n"
                "ON CONFLICT(k) DO UPDATE SET ts=excluded.ts, english=excluded.english, marathi=excluded.marathi",
                [key, now, json.dumps(english, ensure_ascii=False), json.dumps(marathi, ensure_ascii=False)],
            )
            con.commit()
        finally:
            con.close()
    except Exception:
        pass

def _generate_name_variations_llm(input_name: str) -> dict:
	"""
	Use LLM to generate realistic name variations in both English and Marathi.
	Returns dict with 'english_variants' and 'marathi_variants' lists.
	"""
    if not input_name:
        return {"english_variants": [], "marathi_variants": []}
    # cache lookup by normalized key
    try:
        norm_key = nlp.normalize_name(input_name)
    except Exception:
        norm_key = (input_name or "").strip().lower()
    cached = _cache_get_variants(norm_key)
    if cached:
        return cached
	
	try:
		client = OpenAI(
			base_url="http://192.168.1.198:11434/v1",
			api_key="ollama"
		)

		# Prompt updated per user's specification
		system_prompt = (
			"You are a multilingual transliteration and name variant generator.\n"
			"Return valid JSON only. No prose."
		)
		user_prompt = (
			"You are a multilingual transliteration and name variant generator.\n"
			"You generate all reasonable Marathi and English name variations of a given name.\n\n"
			"Rules:\n"
			"1. Do NOT add or remove letters that change pronunciation unnaturally.\n"
			"2. Preserve original structure; only minor phonetic spelling variations.\n"
			"3. Avoid adding 'a', 'ha', 'na', or 'nna' endings that change pronunciation.\n"
			"4. Include Marathi vowel/consonant variants naturally used by Marathi speakers.\n"
			"   - all combinations of \n"
			"        'ी': 'ि', 'ि': 'ी', # Velanti\n"
			"        'ू': 'ु', 'ु': 'ू', # Ukar\n"
			"5. Include English transliteration variants with minor phonetic differences.\n"
			"6. Generate short-form and initial-based variations like:\n"
			"   - Initials (L.S.Jain, L. S. Jain)\n"
			"   - Mixed (Lakshmi.S.Jain, L.Suresh.Jain)\n"
			"   - Dotted and spaced forms for Marathi too (ल.स.जैन, लक्ष्मी.स.जैन)\n"
			"7. Ensure both English and Marathi variants have:\n"
			"   - Normal full names\n"
			"   - Abbreviated names (with initials)\n"
			"   - Punctuated forms (with or without spaces)\n"
			"8. Return valid JSON only, with two keys: \"marathi_variations\" and \"english_variations\".\n\n"
			f"Name: {input_name}\n\n"
			"Example format:\n"
			"{\n"
			"  \"marathi_variations\": [\n"
			"    \"लक्ष्मी सुरेश जैन\", \"लक्स्मी सुरेश जैन\", \"ल.स.जैन\", \"लक्ष्मी.स.जैन\"\n"
			"  ],\n"
			"  \"english_variations\": [\n"
			"    \"Lakshmi Suresh Jain\", \"Laxmi Suresh Jain\", \"L.S.Jain\", \"L. S. Jain\", \"Lakshmi.S.Jain\"\n"
			"  ]\n"
			"}"
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
			logger.warning("LLM returned empty content for name variations")
			return {"english_variants": [], "marathi_variants": []}
		
		# Try to parse JSON from response
		result = None
		try:
			result = json.loads(text)
		except Exception:
			# Try to extract JSON from markdown code blocks or other wrappers
			json_match = re.search(r"\{[\s\S]*\"(?:marathi_variations|english_variations|variations|english_variants|marathi_variants)\"[\s\S]*\}", text)
			if json_match:
				try:
					result = json.loads(json_match.group(0))
				except Exception:
					pass

		if result:
			# Support multiple shapes:
			# 1) New: { marathi_variations: [], english_variations: [] }
			# 2) Old nested: { variations: { english: [], marathi: [] } }
			# 3) Legacy: { english_variants: [], marathi_variants: [] }
			if "marathi_variations" in result or "english_variations" in result:
				english_variants = result.get("english_variations", [])
				marathi_variants = result.get("marathi_variations", [])
			elif "variations" in result and isinstance(result["variations"], dict):
				english_variants = result["variations"].get("english", [])
				marathi_variants = result["variations"].get("marathi", [])
			else:
				english_variants = result.get("english_variants", [])
				marathi_variants = result.get("marathi_variants", [])
			# Ensure they're lists and limit to reasonable size
			if isinstance(english_variants, list):
				english_variants = english_variants[:12]
			else:
				english_variants = []
			if isinstance(marathi_variants, list):
				marathi_variants = marathi_variants[:12]
			else:
				marathi_variants = []
            # write cache
            try:
                _cache_put_variants(norm_key, english_variants, marathi_variants)
            except Exception:
                pass
            return {"english_variants": english_variants, "marathi_variants": marathi_variants}
		else:
			logger.warning(f"Could not parse LLM response for name variations: {text[:100]}")
			return {"english_variants": [], "marathi_variants": []}
			
	except Exception as e:
		logger.exception("LLM name variation generation failed: %s", e)
		# Fallback to simple phonetic generation on error
        try:
            vars = nlp.generate_all_name_variations(input_name)
            eng = list(vars.get("english", []))[:12]
            mar = list(vars.get("marathi", []))[:12]
            try:
                _cache_put_variants(norm_key, eng, mar)
            except Exception:
                pass
            return {"english_variants": eng, "marathi_variants": mar}
        except Exception:
            return {"english_variants": [], "marathi_variants": []}


@app.get("/phonetics")
def phonetics(seed_name: str = Query(..., description="Name to generate phonetic/variation suggestions for")) -> dict:
	if not seed_name:
		raise HTTPException(status_code=400, detail="seed_name required")
	
	variations = _generate_name_variations_llm(seed_name)
	
	return {
		"input": seed_name,
		"variations": {
			"english": set(variations.get("english_variants", [])),
			"marathi": set(variations.get("marathi_variants", [])),
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
		client = OpenAI(
			base_url="http://192.168.1.198:11434/v1",
			api_key="ollama"
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
