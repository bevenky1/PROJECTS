import pandas as pd
import sqlite3
import os
import sys

# Add project root to path to use logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import logger

def ingest_data():
    db_path = "sales.db"
    csv_dir = "Data_CSV"
    
    if not os.path.exists(csv_dir):
        logger.error(f"Directory {csv_dir} not found.")
        return

    conn = sqlite3.connect(db_path)
    logger.info(f"Connected to SQLite database: {db_path}")

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0].lower().replace(" ", "_")
        file_path = os.path.join(csv_dir, csv_file)
        
        try:
            logger.info(f"Loading {csv_file} into table {table_name}...")
            # Try reading with different encodings if default utf-8 fails
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
                
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")

    conn.close()
    logger.info("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
