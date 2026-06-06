import pandas as pd
from sklearn.model_selection import train_test_split

NUM_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'ChargesPerMonth', 'IsNewCustomer', 'HighSpender'
]

CAT_COLS = [
    'Contract', 'InternetService', 'PaymentMethod',
    'TechSupport', 'OnlineSecurity', 'gender',
    'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'OnlineBackup',
    'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling'
]

FEATURE_COLS = NUM_COLS + CAT_COLS
TARGET_COL = 'Churn'


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw CSV and fix known data issues."""
    df = pd.read_csv(filepath)

    # Fix TotalCharges — spaces instead of NaN for new customers
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode target: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop customerID — not a feature
    df.drop(columns=['customerID'], inplace=True)

    # Feature engineering
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)
    df['HighSpender'] = (df['MonthlyCharges'] > 70).astype(int)

    return df


def split_and_save(df: pd.DataFrame, out_dir: str = 'data/processed'):
    """Stratified 80/20 split, saved to processed/."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    train_df.to_csv(f'{out_dir}/train.csv', index=False)
    test_df.to_csv(f'{out_dir}/test.csv', index=False)

    print(f'Train: {len(train_df)} rows | Test: {len(test_df)} rows')
    print(f'Churn rate train: {y_train.mean():.2%}')
    return X_train, X_test, y_train, y_test