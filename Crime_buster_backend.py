# crime_buster_backend.py

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

KM_TO_DEG = 0.009  # approx conversion from km to degrees


# ---------------- Helper Functions ----------------


def create_grid(
    df: pd.DataFrame,
    grid_size_km: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide the geographical area into grid cells of given size in km.

    Returns:
        (df_with_grid_cols, grid_summary)
        - df_with_grid_cols: original df with grid_x, grid_y, grid_id
        - grid_summary: per-grid aggregated info including crime_count
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns.")

    grid_size_deg = grid_size_km * KM_TO_DEG

    # Use floor so negative coords behave nicely
    df = df.copy()
    df["grid_x"] = np.floor(df["longitude"] / grid_size_deg).astype(int)
    df["grid_y"] = np.floor(df["latitude"] / grid_size_deg).astype(int)
    df["grid_id"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)

    # Base summary
    grid_summary = (
        df.groupby("grid_id")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            grid_x=("grid_x", "first"),
            grid_y=("grid_y", "first"),
        )
        .reset_index()
    )

    # Crime count per grid
    crime_counts = df.groupby("grid_id").size().rename("crime_count")
    grid_summary = (
        grid_summary.merge(crime_counts, on="grid_id", how="left")
        .fillna({"crime_count": 0})
    )

    return df, grid_summary


def train_and_predict_for_grids(
    df: pd.DataFrame,
    min_grid: float,
    max_grid: float,
    step_grid: float,
) -> Dict[float, Dict[str, Any]]:
    """
    Train and predict hotspots for multiple grid sizes.

    Returns:
        {
          grid_size_km: {
            "hotspots": pd.DataFrame,
            "auc": float
          },
          ...
        }
    """
    results: Dict[float, Dict[str, Any]] = {}

    # Make sure input is clean
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    grid_sizes = np.arange(min_grid, max_grid + 1e-9, step_grid)
    grid_sizes = np.round(grid_sizes, 2)  # nice keys

    for grid_size in grid_sizes:
        df_grid, grid_summary = create_grid(df, float(grid_size))

        X = grid_summary[["crime_count"]].values
        # Label grids as "hot" if above median crime_count
        median_count = grid_summary["crime_count"].median()
        y = (grid_summary["crime_count"] > median_count).astype(int)

        if len(np.unique(y)) < 2:
            # Edge case: all 0 or all 1; no real classification possible
            auc_score = 0.5
            max_count = max(grid_summary["crime_count"].max(), 1)
            grid_summary["risk_score"] = (
                grid_summary["crime_count"] / max_count
            )
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                y_pred = model.predict_proba(X_test)[:, 1]
                auc_score = float(roc_auc_score(y_test, y_pred))

                grid_summary["risk_score"] = model.predict_proba(X)[:, 1]
            except Exception:
                # Fallback: simple normalized crime frequency
                auc_score = 0.5
                max_count = max(grid_summary["crime_count"].max(), 1)
                grid_summary["risk_score"] = (
                    grid_summary["crime_count"] / max_count
                )

        results[float(grid_size)] = {
            "hotspots": grid_summary.sort_values(
                by="risk_score", ascending=False
            ).reset_index(drop=True),
            "auc": auc_score,
        }

    return results


# ---------------- Police-only Stats ----------------


def generate_police_stats(
    df: pd.DataFrame,
    hotspots: pd.DataFrame,
    crime_type_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate extra statistics for police/admin users.

    Returns a dict with (some keys may be missing if not applicable):
      - "crime_counts": DataFrame
      - "timeseries": DataFrame
      - "hotspot_summary": Dict
      - "detailed_hotspots": DataFrame
    """
    stats: Dict[str, Any] = {}

    df = df.copy()

    # Crime counts by type
    if crime_type_col and crime_type_col in df.columns:
        counts = (
            df[crime_type_col]
            .dropna()
            .value_counts()
            .reset_index()
        )
        counts.columns = [crime_type_col, "count"]
        stats["crime_counts"] = counts

    # Crimes over time (daily)
    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
            ts = (
                df.set_index("datetime")
                .resample("D")
                .size()
                .rename("count")
                .reset_index()
            )
            stats["timeseries"] = ts
        except Exception:
            stats["timeseries"] = pd.DataFrame()

    # Hotspot summary + enriched hotspots
    if hotspots is not None and not hotspots.empty:
        stats["hotspot_summary"] = {
            "n_hotspots": int(len(hotspots)),
            "mean_risk": float(hotspots["risk_score"].mean()),
            "median_risk": float(hotspots["risk_score"].median()),
            "max_risk": float(hotspots["risk_score"].max()),
        }

        # For performance, you could limit this to top N, but we'll keep it full
        def nearby_count(row) -> int:
            lat = row["latitude"]
            lon = row["longitude"]
            # ~1km box (roughly) – adjust as needed
            mask = (
                (df["latitude"].between(lat - 0.01, lat + 0.01))
                & (df["longitude"].between(lon - 0.01, lon + 0.01))
            )
            return int(mask.sum())

        enriched = hotspots.copy()
        enriched["nearby_crime_count"] = enriched.apply(
            nearby_count, axis=1
        )
        stats["detailed_hotspots"] = enriched

    return stats


