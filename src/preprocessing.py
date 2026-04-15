import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def preprocess(df):
    # Drop non-useful columns
    X = df.drop(columns=["label", "name"], errors="ignore")
    y = df["label"].values

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Scale features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    return X, y