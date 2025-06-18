import pandas as pd

json_paths = ["AIME-TTT/test.json", "AIME-TTT/train.json"]

for json_path in json_paths:
    parquet_path = json_path.replace(".json", ".parquet")
    df = pd.read_json(json_path)
    df.to_parquet(parquet_path)
