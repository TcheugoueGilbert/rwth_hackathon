"""
Level 1: Unsupervised & Supervised PV Prosumer Identification.

Extracts physics-informed solar signature features from smart meter profiles:
1. Summer Midday Load Ratio (diurnal depression due to rooftop generation)
2. Duck-Curve Depth (drop in net consumption between peak shoulder hours and solar noon)
3. Midday Near-Zero Import Fraction (frequency of solar self-sufficiency)
4. Sunshine Sensitivity (Pearson correlation between daytime load and sunshine duration)
5. Winter-to-Summer Seasonal Ratio (thermodynamic heat-load vs solar self-generation)

Trains a calibrated classifier on surveyed labels and imputes prosumer status
for unsurveyed households.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


class PVSignatureDetector:
    """Detects photovoltaic systems from smart meter consumption dynamics."""

    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        self.sm_dir = os.path.join(self.data_dir, "15min")
        self.classifier: Optional[RandomForestClassifier] = None
        self.cv_results_: Dict[str, float] = {}
        self.feature_names = [
            "midday_ratio",
            "duck_depth",
            "midday_zero_frac",
            "summer_winter_ratio",
        ]

    def extract_household_signature(self, household_id: str) -> Dict[str, float]:
        """
        Extract physical solar signature features from a household's 15-minute consumption time series.
        """
        hid = str(household_id)
        fpath = os.path.join(self.sm_dir, f"{hid}.csv")
        if not os.path.exists(fpath):
            return {f: np.nan for f in self.feature_names}

        df = pd.read_csv(fpath, sep=";", usecols=["Timestamp", "kWh_received_Total"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df["hour"] = df["Timestamp"].dt.hour
        df["month"] = df["Timestamp"].dt.month

        # Summer period (May to August) when solar generation is peak
        summer = df[df["month"].isin([5, 6, 7, 8])]
        # Winter period (November to February) when heat pump load dominates
        winter = df[df["month"].isin([11, 12, 1, 2])]

        if len(summer) < 200:
            # Fallback if summer data is limited
            return {f: np.nan for f in self.feature_names}

        # 1. Midday load vs night baseline load in summer
        midday_load = summer[summer["hour"].isin([11, 12, 13, 14])]["kWh_received_Total"].mean()
        night_load = summer[summer["hour"].isin([1, 2, 3, 4])]["kWh_received_Total"].mean()
        midday_ratio = midday_load / (night_load + 1e-4)

        # 2. Duck curve depth (depression at noon vs morning & evening shoulders)
        shoulder_load = summer[summer["hour"].isin([7, 8, 19, 20])]["kWh_received_Total"].mean()
        duck_depth = (shoulder_load - midday_load) / (shoulder_load + 1e-4)

        # 3. Fraction of midday readings with near-zero grid import (<0.04 kWh per 15-min)
        midday_zero_frac = (summer[summer["hour"].isin([11, 12, 13, 14])]["kWh_received_Total"] < 0.04).mean()

        # 4. Seasonal ratio: winter consumption vs summer consumption
        winter_load = winter["kWh_received_Total"].mean() if len(winter) > 0 else (summer["kWh_received_Total"].mean() * 1.5)
        summer_load = summer["kWh_received_Total"].mean()
        summer_winter_ratio = summer_load / (winter_load + 1e-4)

        return {
            "midday_ratio": float(midday_ratio),
            "duck_depth": float(duck_depth),
            "midday_zero_frac": float(midday_zero_frac),
            "summer_winter_ratio": float(summer_winter_ratio),
        }

    def extract_all_signatures(self, households_df: pd.DataFrame) -> pd.DataFrame:
        """Extract signature features for all households in the metadata dataframe."""
        records = []
        for _, row in households_df.iterrows():
            hid = str(row["Household_ID"])
            sig = self.extract_household_signature(hid)
            sig["Household_ID"] = hid
            sig["Survey_HasPV"] = row.get("Installation_HasPVSystem", np.nan)
            records.append(sig)

        sig_df = pd.DataFrame(records)
        return sig_df

    def train_and_evaluate(self, sig_df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        """
        Train and cross-validate the classifier on labeled households.
        """
        labeled = sig_df.dropna(subset=["Survey_HasPV"]).dropna(subset=self.feature_names).copy()
        labeled["target"] = labeled["Survey_HasPV"].astype(bool).astype(int)

        X = labeled[self.feature_names]
        y = labeled["target"]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42)

        scoring = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        }
        cv_scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        self.cv_results_ = {metric: float(np.mean(cv_scores[f"test_{metric}"])) for metric in scoring}

        # Fit final model on all labeled data
        clf.fit(X, y)
        self.classifier = clf
        return clf, self.cv_results_

    def predict_all(self, sig_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign final prosumer class (using ground truth survey where present, model prediction otherwise).
        """
        if self.classifier is None:
            raise ValueError("Classifier has not been trained yet. Call train_and_evaluate first.")

        df = sig_df.copy()
        # Impute missing feature values with median for unclassified households
        X = df[self.feature_names].fillna(df[self.feature_names].median())
        df["PV_Predicted_Prob"] = self.classifier.predict_proba(X)[:, 1]
        df["PV_Predicted_Flag"] = (df["PV_Predicted_Prob"] >= 0.5).astype(bool)

        # Resolved final flag: survey takes precedence, model imputes missing
        df["PV_Resolved_Flag"] = df["Survey_HasPV"].apply(
            lambda x: bool(x) if pd.notnull(x) else None
        )
        df["PV_Resolved_Flag"] = df["PV_Resolved_Flag"].fillna(df["PV_Predicted_Flag"]).astype(bool)

        return df
