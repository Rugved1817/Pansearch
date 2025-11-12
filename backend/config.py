from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Columns:
	pan: str = "pan_numbers"
	name: str = "buyer"
	alt_name: str = "seller"
	age: str = "age"
	address: str = "address"
	mobile: str = "mobile"


@dataclass(frozen=True)
class Thresholds:
	name_strict: int = 90
	name_medium: int = 82
	name_loose: int = 75
	address_loose: int = 72
	address_medium: int = 80
	mobile_exact_required: bool = False


@dataclass(frozen=True)
class SearchParams:
	candidate_limit: int = 20000
	max_per_phonetic_key: int = 5000
	limit_return_rows: Optional[int] = None


COLUMNS = Columns()
THRESHOLDS = Thresholds()
SEARCH = SearchParams()
