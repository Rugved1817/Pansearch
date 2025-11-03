import argparse
import os
import duckdb


def main(parquet_dir: str, db_path: str) -> None:
	os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
	con = duckdb.connect(db_path)
	# Create physical table for faster indexing vs querying raw parquet every time
	con.execute("DROP TABLE IF EXISTS transactions")
	con.execute(
		"""
		CREATE TABLE transactions AS
		SELECT * FROM read_parquet(?);
		""",
		[os.path.join(parquet_dir, "*.parquet")],
	)
	# Lightweight indexes
	con.execute("CREATE INDEX IF NOT EXISTS idx_pan ON transactions(pan_upper)")
	con.execute("CREATE INDEX IF NOT EXISTS idx_name_phonetic ON transactions(name_phonetic)")
 
	con.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
 
	parser.add_argument("--parquet", required=True, help="Directory containing parquet files")
	parser.add_argument("--db", required=True, help="DuckDB file path to create")
	args = parser.parse_args()
	main(args.parquet, args.db)
