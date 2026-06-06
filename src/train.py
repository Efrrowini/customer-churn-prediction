import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

from src.preprocess import (
    load_and_clean, split_and_save,
    NUM_COLS, CAT_COLS, TARGET_COL
)

RAW_PATH = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_PATH = 'models/churn_pipeline.pkl'


def build_pipeline() -> Pipeline:

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ), CAT_COLS)
        ],
        remainder='drop'
    )

    model = XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        eval_metric='auc',
        random_state=42,
        verbosity=0
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def train():
    print('Loading and cleaning data...')
    df = load_and_clean(RAW_PATH)

    # Full dataset for cross-validation
    X_all = df[NUM_COLS + CAT_COLS]
    y_all = df[TARGET_COL]

    # Fixed split for final evaluation
    X_train, X_test, y_train, y_test = split_and_save(df)

    pipeline = build_pipeline()

    # Cross-validation AUC — this is the real number to report
    print('Running 5-fold cross-validation...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_all, y_all,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    print(f'CV AUC scores: {[round(s,4) for s in cv_scores]}')
    print(f'CV AUC mean : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

    mlflow.set_experiment('churn-prediction')

    with mlflow.start_run(run_name='xgboost-cv'):

        print('Training final pipeline on full train set...')
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        auc  = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        mlflow.log_params({
            'n_estimators': 800,
            'max_depth': 3,
            'learning_rate': 0.02,
            'cv_folds': 5,
        })
        mlflow.log_metrics({
            'auc_roc_test': auc,
            'auc_roc_cv_mean': cv_scores.mean(),
            'auc_roc_cv_std': cv_scores.std(),
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        mlflow.sklearn.log_model(pipeline, 'model')

        print(f'\n{"="*40}')
        print(f'  CV AUC    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
        print(f'  Test AUC  : {auc:.4f}')
        print(f'  Precision : {prec:.4f}')
        print(f'  Recall    : {rec:.4f}')
        print(f'  F1-score  : {f1:.4f}')
        print(f'{"="*40}')
        print(classification_report(y_test, y_pred,
              target_names=['No Churn', 'Churn']))

    joblib.dump(pipeline, MODEL_PATH)
    print(f'\nModel saved -> {MODEL_PATH}')


if __name__ == '__main__':
    train()