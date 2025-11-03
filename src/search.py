import argparse
from collections import Counter
from typing import Iterable, List, Optional, Tuple, Dict

import duckdb

try:
	from .config import COLUMNS as COLS, THRESHOLDS as TH, SEARCH as SP  # type: ignore
	from .utils import nlp  # type: ignore
except ImportError:  # script execution fallback
	from config import COLUMNS as COLS, THRESHOLDS as TH, SEARCH as SP
	from utils import nlp


def pick_canonical_names(rows: List[Tuple[str]]) -> List[str]:
	# rows are tuples containing name_norm
	counter = Counter([r[0] for r in rows if r and r[0]])
	return [n for n, _ in counter.most_common(3)]


def _get_table_columns(con: duckdb.DuckDBPyConnection, table: str = "transactions") -> set:
	rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
	# rows: (cid, name, type, notnull, dflt_value, pk)
	return {r[1] for r in rows}


def fetch_rows_for_pan(con: duckdb.DuckDBPyConnection, pan: str) -> List[Tuple[str, str, str, str, str]]:
	pan_up = nlp.canonicalize_pan(pan)
	cols = _get_table_columns(con)
	select_parts: List[str] = [
		"name_norm",
		("name_phonetic" if "name_phonetic" in cols else "'' AS name_phonetic"),
		("address" if "address" in cols else "'' AS address"),
		("mobile" if "mobile" in cols else "'' AS mobile"),
		("pan_upper" if "pan_upper" in cols else "'' AS pan_upper"),
	]
	q = f"SELECT {', '.join(select_parts)} FROM transactions WHERE pan_upper = ?"
	return con.execute(q, [pan_up]).fetchall()


def candidate_block_by_names(con: duckdb.DuckDBPyConnection, names: List[str]) -> List[Tuple]:
	cols = _get_table_columns(con)
	# Prefer phonetic blocking if available
	if "name_phonetic" in cols:
		keys = list({nlp.phonetic_key(n) for n in names if n})
		if not keys:
			return []
		placeholders = ",".join(["?"] * len(keys))
		select_parts: List[str] = [
			"name_norm",
			"name_phonetic",
			("address" if "address" in cols else "'' AS address"),
			("mobile" if "mobile" in cols else "'' AS mobile"),
			("pan_upper" if "pan_upper" in cols else "'' AS pan_upper"),
		]
		q = f"""
			SELECT {', '.join(select_parts)}
			FROM transactions
			WHERE name_phonetic IN ({placeholders})
			LIMIT {SP.candidate_limit}
		"""
		return con.execute(q, keys).fetchall()
	# Fallback: use exact name_norm blocking (less recall but works without phonetics)
	norms = list({nlp.normalize_name(n) for n in names if n})
	if not norms:
		return []
	placeholders = ",".join(["?"] * len(norms))
	select_parts = [
		"name_norm",
		("'' AS name_phonetic"),
		("address" if "address" in cols else "'' AS address"),
		("mobile" if "mobile" in cols else "'' AS mobile"),
		("pan_upper" if "pan_upper" in cols else "'' AS pan_upper"),
	]
	q = f"""
		SELECT {', '.join(select_parts)}
		FROM transactions
		WHERE name_norm IN ({placeholders})
		LIMIT {SP.candidate_limit}
	"""
	return con.execute(q, norms).fetchall()


def candidate_block_by_name_like(con: duckdb.DuckDBPyConnection, seed_name: str) -> List[Tuple]:
	"""
	Fallback blocking using LIKE on normalized tokens when phonetic blocking misses.
	"""
	cols = _get_table_columns(con)
	seed_norm = nlp.normalize_name(seed_name)
	if not seed_norm:
		return []
	# Build LIKE patterns for each token (require at least one token match)
	tokens = [t for t in set(seed_norm.split(" ")) if t]
	if not tokens:
		return []
	# AND across tokens for stricter match
	clauses = ["name_norm LIKE ?" for _ in tokens]
	params = [f"%{t}%" for t in tokens]
	select_parts: List[str] = [
		"name_norm",
		("name_phonetic" if "name_phonetic" in cols else "'' AS name_phonetic"),
		("address" if "address" in cols else "'' AS address"),
		("mobile" if "mobile" in cols else "'' AS mobile"),
		("pan_upper" if "pan_upper" in cols else "'' AS pan_upper"),
	]
	q = f"""
		SELECT {', '.join(select_parts)}
		FROM transactions
		WHERE {' AND '.join(clauses)}
		LIMIT {SP.candidate_limit}
	"""
	return con.execute(q, params).fetchall()


def score_candidate(base_names: List[str], row: Tuple[str, str, str, str, str]) -> float:
	name_norm, _phon, address, mobile, _pan = row
	if not name_norm:
		return 0.0
	best_name = max((nlp.fuzzy_name_score(name_norm, b) for b in base_names), default=0)
	score = best_name / 100.0
	# Light boosts for corroborating attributes
	if best_name < TH.name_strict:
		addr_score = nlp.fuzzy_address_score(address or "", address or "") if address else 0
		if addr_score >= TH.address_loose:
			score = min(1.0, score + 0.05)
		if TH.mobile_exact_required and mobile:
			score = min(1.0, score + 0.05)
	return max(0.0, min(1.0, score))


def expand_all_transactions_for_entities(con: duckdb.DuckDBPyConnection, pans: List[str], names: List[str]) -> List[Tuple]:
	pans = [nlp.canonicalize_pan(p) for p in pans if p]
	names = list({n for n in names if n})
	conds = []
	params: List[str] = []
	if pans:
		placeholders = ",".join(["?"] * len(pans))
		conds.append(f"pan_upper IN ({placeholders})")
		params.extend(pans)
	if names:
		placeholders = ",".join(["?"] * len(names))
		conds.append(f"name_norm IN ({placeholders})")
		params.extend(names)
	where = " OR ".join(conds) if conds else "1=0"
	q = f"SELECT * FROM transactions WHERE {where}"
	if SP.limit_return_rows:
		q += f" LIMIT {SP.limit_return_rows}"
	return con.execute(q, params).fetchall()


def expand_all_transactions_with_scores(
	con: duckdb.DuckDBPyConnection,
	pans: List[str],
	name_to_score: Dict[str, float],
) -> List[Tuple[Tuple, float]]:
	pans = [nlp.canonicalize_pan(p) for p in pans if p]
	names = list({n for n in name_to_score.keys() if n})
	conds = []
	params: List[str] = []
	if pans:
		placeholders = ",".join(["?"] * len(pans))
		conds.append(f"pan_upper IN ({placeholders})")
		params.extend(pans)
	if names:
		placeholders = ",".join(["?"] * len(names))
		conds.append(f"name_norm IN ({placeholders})")
		params.extend(names)
	where = " OR ".join(conds) if conds else "1=0"
	q = f"SELECT * FROM transactions WHERE {where}"
	if SP.limit_return_rows:
		q += f" LIMIT {SP.limit_return_rows}"
	rows = con.execute(q, params).fetchall()
	# Attach scores: exact PAN hits get 1.0; name hits use name_to_score
	# We need column positions for name_norm and pan_upper
	cols = [r[1] for r in con.execute("PRAGMA table_info('transactions')").fetchall()]
	try:
		idx_name = cols.index("name_norm")
	except ValueError:
		idx_name = -1
	try:
		idx_pan = cols.index("pan_upper")
	except ValueError:
		idx_pan = -1
	out: List[Tuple[Tuple, float]] = []
	for row in rows:
		score = 0.0
		if idx_pan >= 0 and row[idx_pan]:
			if row[idx_pan] in pans:
				score = 1.0
		if score < 1.0 and idx_name >= 0 and row[idx_name]:
			score = max(score, float(name_to_score.get(row[idx_name], 0.0)))
		out.append((row, score))
	return out


def search_by_pan(con: duckdb.DuckDBPyConnection, pan: str) -> List[Tuple]:
	base_rows = fetch_rows_for_pan(con, pan)
	if not base_rows:
		return []
	base_names = pick_canonical_names([(r[0],) for r in base_rows])
	candidates = candidate_block_by_names(con, base_names)
	# Score candidates and keep non-zero
	name_to_score: Dict[str, float] = {}
	for r in candidates:
		sc = score_candidate(base_names, r)
		if sc > 0:
			# Keep max score per name
			nm = r[0]
			name_to_score[nm] = max(name_to_score.get(nm, 0.0), sc)
	# Base PAN rows are exact matches -> 1.0
	all_pans = {r[4] for r in base_rows if r[4]}
	for r in base_rows:
		nm = r[0]
		if nm:
			name_to_score[nm] = max(1.0, name_to_score.get(nm, 0.0))
	rows_with_scores = expand_all_transactions_with_scores(con, list(all_pans), name_to_score)
	# Return just rows; API will attach scores
	return [rw for rw, _ in rows_with_scores]


def search_by_seed_name(con: duckdb.DuckDBPyConnection, seed_name: str) -> List[Tuple]:
	seed_norm = nlp.normalize_name(seed_name)
	# Generate variations to improve recall
	var_dict = nlp.generate_all_name_variations(seed_name)
	all_variations = {seed_norm}
	all_variations.update(var_dict.get("marathi", set()))
	all_variations.update(var_dict.get("english", set()))
	# Normalize all variations
	base_names = [nlp.normalize_name(v) for v in all_variations if v]
	base_names = list({b for b in base_names if b})  # dedupe
	
	candidates = candidate_block_by_names(con, base_names)
	if not candidates:
		# Fallback to LIKE-based blocking if phonetic keys don't hit - use OR to get more candidates
		seed_norm_local = nlp.normalize_name(seed_name)
		if seed_norm_local:
			tokens = [t for t in set(seed_norm_local.split(" ")) if len(t) >= 2]
			if tokens:
				cols = _get_table_columns(con)
				clauses = ["name_norm LIKE ?" for _ in tokens]
				params = [f"%{t}%" for t in tokens]
				select_parts: List[str] = [
					"name_norm",
					("name_phonetic" if "name_phonetic" in cols else "'' AS name_phonetic"),
					("address" if "address" in cols else "'' AS address"),
					("mobile" if "mobile" in cols else "'' AS mobile"),
					("pan_upper" if "pan_upper" in cols else "'' AS pan_upper"),
				]
				# Use OR initially to get more candidates, we'll filter later
				q = f"""
					SELECT {', '.join(select_parts)}
					FROM transactions
					WHERE {' OR '.join(clauses)}
					LIMIT {SP.candidate_limit}
				"""
				candidates = con.execute(q, params).fetchall()
	
	name_to_score: Dict[str, float] = {}
	# Build strict token set (length>=2) for devanagari/latin alike
	seed_tokens = {t for t in seed_norm.split(" ") if len(t) >= 2}
	for r in candidates:
		sc = score_candidate(base_names, r)
		# Check token overlap - require at least 2 tokens match for multi-token names
		cand_tokens = {t for t in (r[0] or "").split(" ") if t and len(t) >= 2}
		token_overlap = len(seed_tokens & cand_tokens)
		# More lenient: accept if score is decent OR if most tokens match
		most_tokens_ok = len(seed_tokens) > 0 and token_overlap >= max(1, len(seed_tokens) - 1)
		strict_ok = seed_tokens.issubset(cand_tokens)
		
		if sc >= (TH.name_loose / 100.0) or strict_ok or most_tokens_ok:
			name_to_score[r[0]] = max(name_to_score.get(r[0], 0.0), sc)
	
	# Include the seed name variations
	for nm in base_names:
		if nm:
			name_to_score[nm] = max(name_to_score.get(nm, 0.0), 0.95)
	
	pans = [r[4] for r in candidates if r[4]]
	rows_with_scores = expand_all_transactions_with_scores(con, pans, name_to_score)
	return [rw for rw, _ in rows_with_scores]


def main(db_path: str, pan: Optional[str], seed_name: Optional[str]) -> None:
	con = duckdb.connect(db_path, read_only=True)
	if pan:
		rows = search_by_pan(con, pan)
	elif seed_name:
		rows = search_by_seed_name(con, seed_name)
	else:
		raise SystemExit("Provide either --pan or --seed-name")
	print(f"Rows: {len(rows)}")
	# Print a small sample
	for row in rows[:20]:
		print(row)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--db", required=True, help="Path to DuckDB database")
	parser.add_argument("--pan", help="PAN to search")
	parser.add_argument("--seed-name", help="Seed name (Marathi or English)")
	args = parser.parse_args()
	main(args.db, args.pan, args.seed_name)
