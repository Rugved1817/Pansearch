import re
import unicodedata
from typing import Iterable, List, Optional, Tuple, Dict, Set

import phonetics
from rapidfuzz import fuzz
from unidecode import unidecode
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

DEVANAGARI_BLOCK = (0x0900, 0x097F)
PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", re.IGNORECASE)


def is_devanagari(text: str) -> bool:
	if not text:
		return False
	for ch in text:
		cp = ord(ch)
		if DEVANAGARI_BLOCK[0] <= cp <= DEVANAGARI_BLOCK[1]:
			return True
	return False


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text or "").strip()


def normalize_punct(text: str) -> str:
	return re.sub(r"[^\w\s\u0900-\u097F]", " ", text or "")


def to_nfc(text: str) -> str:
	return unicodedata.normalize("NFC", text or "")


def normalize_name(raw: Optional[str]) -> str:
	if not raw:
		return ""
	text = to_nfc(raw)
	text = normalize_punct(text)
	text = normalize_whitespace(text)
	return text.lower()


def detect_language(text: str) -> str:
	return "devanagari" if is_devanagari(text) else "latin"


def devanagari_to_latin(text: str) -> str:
	if not text:
		return ""
	return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)


def english_phonetic_key(text: str) -> str:
	p, s = phonetics.dmetaphone(text or "")
	return (p or s or "")


def marathi_phonetic_key(text: str) -> str:
	latin = devanagari_to_latin(text)
	p, s = phonetics.dmetaphone(latin or "")
	return (p or s or "")


def phonetic_key(text: str) -> str:
	if not text:
		return ""
	lang = detect_language(text)
	return marathi_phonetic_key(text) if lang == "devanagari" else english_phonetic_key(text)


def fuzzy_name_score(a: str, b: str) -> int:
	return int(fuzz.token_set_ratio(a, b))


def fuzzy_address_score(a: str, b: str) -> int:
	return int(fuzz.token_set_ratio(a, b))


def canonicalize_pan(pan: Optional[str]) -> str:
	if not pan:
		return ""
	return re.sub(r"\s+", "", pan).upper()


def extract_pan_codes(blob: Optional[str]) -> List[str]:
	if not blob:
		return []
	return [canonicalize_pan(m.group(1)) for m in PAN_REGEX.finditer(blob)]

NAME_SNIPPET_PATTERNS = [
	re.compile(r"नाव[:-]\s*([^;:,\n]+?)\s*(?:वय|पत्ता|पॅन|,|;|\n)")
]


def extract_names_from_blob(blob: Optional[str]) -> List[str]:
	if not blob:
		return []
	candidates: List[str] = []
	for pat in NAME_SNIPPET_PATTERNS:
		candidates.extend([normalize_name(m.group(1)) for m in pat.finditer(blob)])
	if not candidates:
		parts = re.split(r"\b\d+\)\s*[:：]", blob)
		for p in parts:
			p = normalize_whitespace(p)
			if p:
				words = re.split(r"\s+", normalize_punct(p))
				if 1 <= len(words) <= 6:
					candidates.append(normalize_name(" ".join(words)))
	seen = set()
	uniq = []
	for c in candidates:
		if c and c not in seen:
			seen.add(c)
			uniq.append(c)
	return uniq[:3]


def _generate_simple_variations(text: str) -> List[str]:
	if not text:
		return []
	variations: Set[str] = {text}
	
	vowel_pairs = [
		("i", "ee"), ("ee", "i"),
		("u", "oo"), ("oo", "u"),
		("a", "aa"), ("aa", "a"),
	]
	
	for old_vowel, new_vowel in vowel_pairs:
		if old_vowel in text:
			variations.add(text.replace(old_vowel, new_vowel, 1))
			if text.rfind(old_vowel) != text.find(old_vowel):
				variations.add(text.rsplit(old_vowel, 1)[0] + new_vowel + text.rsplit(old_vowel, 1)[1])
	
	consonant_pairs = [
		("v", "w"), ("w", "v"),
		("j", "z"), ("z", "j"),
		("s", "sh"), ("sh", "s"),
	]
	
	for old_cons, new_cons in consonant_pairs:
		if old_cons in text:
			variations.add(text.replace(old_cons, new_cons, 1))
	
	return sorted(list(variations))[:8]


def generate_all_name_variations(input_name: str) -> Dict[str, Set[str]]:
	if not input_name:
		return {"marathi": set(), "english": set()}
	
	base_english = ""
	base_devanagari = ""
	
	if is_devanagari(input_name):
		base_devanagari = input_name
		try:
			base_english = transliterate(input_name, sanscript.DEVANAGARI, sanscript.ITRANS)
		except Exception:
			base_english = input_name.lower()
	else:
		base_english = (input_name or "").lower()
		try:
			base_devanagari = transliterate(base_english, sanscript.ITRANS, sanscript.DEVANAGARI)
		except Exception:
			base_devanagari = input_name
	
	english_vars = _generate_simple_variations(base_english)
	english_variations: Set[str] = set(english_vars)
	
	marathi_variations: Set[str] = {base_devanagari}
	
	marathi_swaps = [
		("ी", "ि"), ("ि", "ी"),
		("ू", "ु"), ("ु", "ू"),
		("श", "स"), ("स", "श"),
	]
	
	for old_char, new_char in marathi_swaps:
		if old_char in base_devanagari:
			marathi_variations.add(base_devanagari.replace(old_char, new_char, 1))
	
	marathi_variations = set(list(marathi_variations)[:6])
	
	for marathi_var in list(marathi_variations)[:3]:
		try:
			english_variations.add(transliterate(marathi_var, sanscript.DEVANAGARI, sanscript.ITRANS))
		except Exception:
			pass
	
	english_variations = set(list(english_variations)[:8])
	
	return {"marathi": marathi_variations, "english": english_variations}
