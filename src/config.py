from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Columns:
	# Update these to match the CSV
	pan: str = "pan_numbers"  # in transaction.csv
	name: str = "buyer"        # primary name source blob; fallback to seller
	alt_name: str = "seller"   # secondary name source blob
	age: str = "age"           # may not exist; handled gracefully
	address: str = "address"   # may not exist; handled gracefully
	mobile: str = "mobile"     # may not exist; handled gracefully


@dataclass(frozen=True)
class Thresholds:
	# Name fuzzy matching
	name_strict: int = 90   # exact/near-exact
	name_medium: int = 82   # common small variants (reduced)
	name_loose: int = 75    # cross-script/transliteration variants (reduced)

	# Attribute verification
	address_loose: int = 72
	address_medium: int = 80
	mobile_exact_required: bool = False  # set True to require exact mobile for strong verification


@dataclass(frozen=True)
class SearchParams:
	candidate_limit: int = 20000  # upper bound on candidate rows for fuzzy scoring
	max_per_phonetic_key: int = 5000
	limit_return_rows: Optional[int] = None


COLUMNS = Columns()
THRESHOLDS = Thresholds()
SEARCH = SearchParams()
