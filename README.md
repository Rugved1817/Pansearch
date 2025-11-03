## Advanced Person-Centric Search over Large Transaction CSV

This project builds an advanced search system over large transaction data (~20M+ rows), starting from a PAN input and expanding to fuzzy/phonetic name matches in English and Marathi, verifying candidates via attributes (age, address, mobile) and returning all related transactions (including rows missing PAN).

### Features
- PAN lookup to find base transactions and the PAN holder's canonical name
- Dual-script normalization (English + Marathi/Devanagari)
- Phonetic blocking keys (English: Double Metaphone; Marathi: transliteration + metaphone) for scalable candidate generation
- Fuzzy name matching (RapidFuzz) with configurable thresholds
- Attribute verification using age, address, and mobile similarity/overlap
- Columnar storage (Parquet) + DuckDB for fast scans and simple indexing

### Project Layout
- `src/config.py`: Configure column names and thresholds
- `src/utils/nlp.py`: Normalization, transliteration, phonetic keys, fuzzy scoring
- `src/ingest.py`: Convert CSV to Parquet with derived columns for names/phonetics
- `src/build_index.py`: Build DuckDB database and helper indexes
- `src/search.py`: CLI entrypoint implementing the 4-step search workflow

### Install
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell on Windows
pip install -r requirements.txt
```

### Configure
Edit `src/config.py` to match your CSV columns and thresholds. Default expected columns:
- `pan` (string)
- `name` (string, full name)
- `age` (int or string)
- `address` (string)
- `mobile` (string)

### Ingest
Converts CSV to partitioned Parquet and derives helper columns (`name_norm`, `name_lang`, `name_phonetic`):
```bash
python src/ingest.py --csv "path/to/transactions.csv" --out "data/parquet" --rows-per-file 2000000
```

### Build Index
Creates a DuckDB database that references Parquet files and builds helper views:
```bash
python src/build_index.py --parquet "data/parquet" --db "data/tx.duckdb"
```

### Search
Run the 4-step search from a PAN input:
```bash
python src/search.py --db "data/tx.duckdb" --pan "ABCDE1234F"
```
You can also input Marathi PAN holder names by PAN or seed name. For seed-name search:
```bash
python src/search.py --db "data/tx.duckdb" --seed-name "चिरायु संजय गिरी"
```

### Performance Tips
- Use `--rows-per-file` during ingest to balance file sizes (100MB–512MB per Parquet file recommended)
- Place `data/` on a fast SSD
- Adjust `CANDIDATE_LIMIT` and thresholds in `src/config.py`
- Prefer running searches with a PAN when possible; name-only searches are heavier

### Notes
- Double Metaphone is leveraged for English; Marathi is transliterated to Latin before phonetic hashing. We combine phonetic blocking with fuzzy ratios to handle small spelling/orthographic variations.
- For production-scale recall, consider adding a vector store on top of character n-gram embeddings of names and addresses.
