import argparse
import os
from typing import Optional

import polars as pl

from config import COLUMNS as COLS
from utils import nlp


def derive_columns(df: pl.DataFrame) -> pl.DataFrame:
	# Prefer primary name blob, fallback to alt_name
	has_primary = COLS.name in df.columns
	has_alt = getattr(COLS, "alt_name", None) in df.columns if hasattr(COLS, "alt_name") else False
	name_source = (
		pl.when(pl.lit(has_primary)).then(pl.col(COLS.name)).otherwise(pl.lit(None)).alias("_name_primary")
	)
	alt_source = (
		pl.when(pl.lit(has_alt)).then(pl.col(COLS.alt_name)).otherwise(pl.lit(None)).alias("_name_alt")
	)
	df = df.with_columns([name_source, alt_source])

	# Extract candidate names from blobs (buyer first, then seller)
	df = df.with_columns(
		[
			pl.coalesce(
				[
					pl.col("_name_primary").map_elements(lambda s: (nlp.extract_names_from_blob(s) or [""])[0], return_dtype=pl.Utf8),
					pl.col("_name_alt").map_elements(lambda s: (nlp.extract_names_from_blob(s) or [""])[0], return_dtype=pl.Utf8),
				]
			).alias("name_extracted"),
		]
	)

	# Normalize and phonetic key
	df = (
		df.with_columns(
			[
				pl.col("name_extracted").map_elements(nlp.normalize_name, return_dtype=pl.Utf8).alias("name_norm"),
				pl.col("name_extracted").map_elements(nlp.detect_language, return_dtype=pl.Utf8).alias("name_lang"),
				pl.col("name_extracted").map_elements(nlp.phonetic_key, return_dtype=pl.Utf8).alias("name_phonetic"),
			]
		)
	)

	# PAN: use structured column if present; also parse from blobs for better recall
	pan_series = pl.when(pl.lit(COLS.pan in df.columns)).then(pl.col(COLS.pan)).otherwise(pl.lit(None))
	df = df.with_columns([pan_series.alias("pan_raw")])
	df = df.with_columns([
		pl.coalesce([
			pl.col("pan_raw").map_elements(nlp.canonicalize_pan, return_dtype=pl.Utf8),
			pl.col("_name_primary").map_elements(lambda s: (nlp.extract_pan_codes(s) or [""])[0], return_dtype=pl.Utf8),
			pl.col("_name_alt").map_elements(lambda s: (nlp.extract_pan_codes(s) or [""])[0], return_dtype=pl.Utf8),
		]).alias("pan_upper")
	])

	return df


def write_partitioned_parquet(df: pl.LazyFrame, out_dir: str, rows_per_file: int = 2_000_000) -> None:
	os.makedirs(out_dir, exist_ok=True)
	count = df.select(pl.len()).collect().item()
	if count == 0:
		return
	parts = max(1, (count + rows_per_file - 1) // rows_per_file)
	with pl.SQLContext() as ctx:
		ctx.register("t", df)
		for i in range(parts):
			offset = i * rows_per_file
			chunk = ctx.execute(f"SELECT * FROM t LIMIT {rows_per_file} OFFSET {offset}").collect()
			file_path = os.path.join(out_dir, f"part-{i:05d}.parquet")
			chunk.write_parquet(file_path)


def main(csv_path: str, out_dir: str, rows_per_file: int) -> None:
	scan = pl.scan_csv(csv_path, infer_schema_length=20000, ignore_errors=True)
	# Select available columns
	want = [c for c in [COLS.pan, COLS.name, getattr(COLS, "alt_name", None)] if c]
	needed = [c for c in want if c in scan.columns]
	lf = scan.select(needed)
	df = lf.collect()
	df = derive_columns(df)
	write_partitioned_parquet(df.lazy(), out_dir, rows_per_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv", required=True, help="Path to source CSV")
	parser.add_argument("--out", required=True, help="Output directory for Parquet files")
	parser.add_argument("--rows-per-file", type=int, default=2_000_000)
	args = parser.parse_args()
	main(args.csv, args.out, args.rows_per_file)
