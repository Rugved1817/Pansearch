import os
import logging
from typing import List, Optional
import re
import json
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


@app.get("/phonetics")
def phonetics(seed_name: str = Query(..., description="Name to generate phonetic/variation suggestions for")) -> dict:
    if not seed_name:
        raise HTTPException(status_code=400, detail="seed_name required")
    vars = nlp.generate_all_name_variations(seed_name)
    # Also include simple phonetic keys for the bases
    return {
        "input": seed_name,
        "variations": vars,
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
		base_url = os.getenv("LLM_BASE_URL", "http://192.168.1.198:11434/v1")
		api_key = os.getenv("LLM_API_KEY", "ollama")
		model = os.getenv("LLM_MODEL", "gpt-oss:20b")
		client = OpenAI(base_url=base_url, api_key=api_key)
		system_prompt = "Return JSON only. You extract concise person names."
		user_prompt = (
			"Extract only the main human person's name from the text.\n"
			"- Text may include roles (e.g., संचालक, तर्फे), organizations, or punctuation.\n"
			"- Preserve the script (Marathi remains Marathi, English remains English).\n"
			"- Respond ONLY as compact JSON: {\\\"name\\\": \\\"<NAME>\\\"}. No extra text.\n\n"
			f"Text: {raw_name}"
		)
		resp = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=0,
			max_tokens=64,
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
		# LLM extraction for a concise human name
		extracted = _extract_person_name_llm(name or "") if name else None
		final_name = extracted or name
		extraction_error = None if (extracted is not None) else ("llm_extraction_failed" if name else None)
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
