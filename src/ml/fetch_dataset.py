import os
import pandas as pd
import urllib.request

def fetch_and_save():
    urls = [
        "https://raw.githubusercontent.com/gchoi/Dataset/master/credit_risk_dataset.csv",
        "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Credit%20Default.csv"
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, "credit_risk_dataset.csv")
    
    for url in urls:
        print(f"Trying to fetch from: {url}")
        try:
            df = pd.read_csv(url)
            df.to_csv(output_path, index=False)
            print(f"Dataset successfully saved to {output_path}")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            return
        except Exception as e:
            print(f"Failed: {e}")
            
    print("Could not fetch dataset from any URL.")

if __name__ == "__main__":
    fetch_and_save()
