import os
import pandas as pd

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data from the provided file path.
    
    Args:
        file_path (str): Path to the raw CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

def save_backup(df: pd.DataFrame, backup_path: str):
    """
    Save a backup copy of the raw data to an interim location.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        backup_path (str): Path to save the backup CSV.
    """
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    df.to_csv(backup_path, index=False)
    print(f"Backup saved to {backup_path}")

if __name__ == "__main__":
    input_path = "data/raw/german_credit_data.csv"
    backup_path = "data/interim/german_credit_data_backup.csv"

    df = load_raw_data(input_path)
    print(f"✅ Data Loaded Successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    save_backup(df, backup_path)
