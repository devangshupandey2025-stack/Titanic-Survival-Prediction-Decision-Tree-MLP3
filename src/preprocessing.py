import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    columns_to_drop = ["name", "ticket", "cabin"]
    df = df.drop(columns=columns_to_drop)
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(
        df["embarked"].mode()[0]
    )
    df["sex"] = df["sex"].map({
        "male": 0,
        "female": 1
    })
    df["embarked"] = df["embarked"].map({
        "S": 0,
        "C": 1,
        "Q": 2
    })
    return df