"""Feature engineering for fraud detection"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class FeatureEngineer:
    """Handle feature engineering for fraud detection"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Engineer features for fraud detection.

        Args:
            df: Input dataframe with transaction data
            fit: Whether to fit the scaler (True for training data)

        Returns:
            DataFrame with engineered features
        """
        # Clean missing and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().copy()

        # Time-based features
        if "Time" in df.columns:
            # Convert time to hours and create cyclical features
            df["hour"] = (df["Time"] / 3600) % 24
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

            # Day of transaction (keep as float, no int casting)
            df["day"] = df["Time"] / 86400.0

        # Amount-based features
        if "Amount" in df.columns:
            # Log transform of amount (add 1 to handle 0 values)
            df["log_amount"] = np.log1p(df["Amount"])

            # Amount buckets (safe integer codes, no NaN -> int crash)
            df["amount_bucket"] = pd.cut(
                df["Amount"],
                bins=[0, 10, 50, 100, 500, 1000, float("inf")],
                labels=False,
                include_lowest=True,
            )
            df["amount_bucket"] = df["amount_bucket"].fillna(0).astype(int)

            # High risk indicators based on domain knowledge
            df["high_amount_flag"] = (
                df["Amount"] > df["Amount"].quantile(0.95)
            ).astype(int)

        # Interaction features between PCA components (most important ones)
        if "V1" in df.columns and "V2" in df.columns:
            df["v1_v2_interaction"] = df["V1"] * df["V2"]

        if "V4" in df.columns and "V11" in df.columns:
            df["v4_v11_interaction"] = df["V4"] * df["V11"]

        return df

    def prepare_features(
        self, df: pd.DataFrame, target_col: str = "Class", fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training/prediction.

        Args:
            df: Input dataframe
            target_col: Name of target column
            fit: Whether to fit preprocessing (True for training)

        Returns:
            Tuple of (features, target)
        """
        # Engineer features
        df_engineered = self.engineer_features(df, fit=fit)

        # Separate target if it exists
        if target_col in df_engineered.columns:
            y = df_engineered[target_col]
            X = df_engineered.drop(columns=[target_col])
        else:
            y = None
            X = df_engineered

        # Store or reuse feature columns
        if fit:
            self.feature_columns = X.columns.tolist()
        else:
            # Ensure columns match training
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_columns]

        # Handle any remaining NaN values
        X = X.fillna(0)

        return X, y

    def get_feature_importance_names(self) -> list:
        """Get names of engineered features"""
        return self.feature_columns if self.feature_columns else []
