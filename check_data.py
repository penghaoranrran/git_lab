import pandas as pd
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        print(f"Successfully loaded {file_path} with GBK")
        return df
    except Exception as e:
        print(f"Failed with GBK: {e}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"Successfully loaded {file_path} with UTF-8")
            return df
        except Exception as e:
            print(f"Failed with UTF-8: {e}")
            return None

if __name__ == "__main__":
    df1 = load_data('Watermelon-train1.csv')
    if df1 is not None:
        print("DF1 Columns:", df1.columns.tolist())
    
    df2 = load_data('Watermelon-train2.csv')
    if df2 is not None:
        print("DF2 Columns:", df2.columns.tolist())
        print("DF2 Head:", df2.head(2))
        print("B2 optimized version")
        print("C4 experimental version")
