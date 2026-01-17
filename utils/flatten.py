import pandas as pd

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if c])
            for col in df.columns
        ]
    else:
        df.columns = df.columns.astype(str)
    return df