import pandas as pd
import sys

def main():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    print(f"Downloading Boston Housing dataset from: {url}")
    try:
        df = pd.read_csv(url)
        df.to_csv("boston_housing.csv", index=False)
        # Also overwrite data.csv to prevent crashes in any baseline scripts
        df.to_csv("data.csv", index=False)
        print("Successfully downloaded and saved dataset as 'boston_housing.csv' and 'data.csv'!")
        print(f"Dataset shape: {df.shape}")
        print("Columns:", list(df.columns))
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
