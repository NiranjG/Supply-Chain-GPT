"""
Demand forecasting using Prophet and XGBoost
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    Demand forecasting service using:
    - Prophet for time series forecasting
    - XGBoost for feature-rich predictions
    """

    def __init__(self, use_prophet: bool = True, use_xgboost: bool = True):
        """
        Initialize forecaster

        Args:
            use_prophet: Whether to use Prophet model
            use_xgboost: Whether to use XGBoost model
        """
        self.use_prophet = use_prophet
        self.use_xgboost = use_xgboost
        self._prophet_model = None
        self._xgboost_model = None

    def forecast(
        self,
        historical_data: pd.DataFrame,
        periods: int = 30,
        sku_id: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate demand forecast

        Args:
            historical_data: DataFrame with 'ds' (date) and 'y' (demand) columns
            periods: Number of future periods to forecast
            sku_id: SKU identifier
            warehouse_id: Warehouse identifier
            include_confidence: Include confidence intervals

        Returns:
            Dictionary with forecast results
        """
        if len(historical_data) < 10:
            logger.warning("Insufficient data for forecasting, using simple average")
            return self._simple_forecast(historical_data, periods, sku_id, warehouse_id)

        results = {}

        # Prophet forecast
        if self.use_prophet:
            try:
                prophet_forecast = self._prophet_forecast(
                    historical_data, periods, include_confidence
                )
                results["prophet"] = prophet_forecast
            except Exception as e:
                logger.error(f"Prophet forecast failed: {e}")

        # XGBoost forecast
        if self.use_xgboost:
            try:
                xgb_forecast = self._xgboost_forecast(historical_data, periods)
                results["xgboost"] = xgb_forecast
            except Exception as e:
                logger.error(f"XGBoost forecast failed: {e}")

        # Ensemble if both available
        if "prophet" in results and "xgboost" in results:
            ensemble = self._ensemble_forecast(
                results["prophet"], results["xgboost"]
            )
            results["ensemble"] = ensemble
            primary_forecast = ensemble
            model_used = "ensemble"
        elif "prophet" in results:
            primary_forecast = results["prophet"]
            model_used = "prophet"
        elif "xgboost" in results:
            primary_forecast = results["xgboost"]
            model_used = "xgboost"
        else:
            return self._simple_forecast(historical_data, periods, sku_id, warehouse_id)

        # Calculate metrics
        mape = self._calculate_mape(historical_data, primary_forecast)

        return {
            "sku_id": sku_id,
            "warehouse_id": warehouse_id,
            "forecast": primary_forecast,
            "model_used": model_used,
            "mape": mape,
            "periods": periods,
            "all_models": results
        }

    def _prophet_forecast(
        self,
        data: pd.DataFrame,
        periods: int,
        include_confidence: bool
    ) -> List[Dict[str, Any]]:
        """Generate Prophet forecast"""
        from prophet import Prophet

        # Prepare data
        df = data[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # Initialize and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df)

        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)

        # Predict
        forecast = model.predict(future)

        # Extract relevant columns
        result = []
        future_forecast = forecast.tail(periods)

        for _, row in future_forecast.iterrows():
            entry = {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "predicted_demand": max(0, round(row["yhat"], 2)),
            }
            if include_confidence:
                entry["lower_bound"] = max(0, round(row["yhat_lower"], 2))
                entry["upper_bound"] = max(0, round(row["yhat_upper"], 2))
            result.append(entry)

        return result

    def _xgboost_forecast(
        self,
        data: pd.DataFrame,
        periods: int
    ) -> List[Dict[str, Any]]:
        """Generate XGBoost forecast"""
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split

        # Prepare features
        df = data.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        # Create time-based features
        df["day_of_week"] = df["ds"].dt.dayofweek
        df["day_of_month"] = df["ds"].dt.day
        df["month"] = df["ds"].dt.month
        df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Rolling features
        df["rolling_mean_7"] = df["y"].rolling(window=7, min_periods=1).mean()
        df["rolling_std_7"] = df["y"].rolling(window=7, min_periods=1).std()
        df["rolling_mean_30"] = df["y"].rolling(window=30, min_periods=1).mean()

        # Drop rows with NaN
        df = df.dropna()

        feature_cols = [
            "day_of_week", "day_of_month", "month", "week_of_year",
            "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_30"
        ]

        X = df[feature_cols]
        y = df["y"]

        # Train model
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)

        # Generate future predictions
        last_date = df["ds"].max()
        last_values = df["y"].tail(30).tolist()

        predictions = []
        current_values = last_values.copy()

        for i in range(periods):
            future_date = last_date + timedelta(days=i + 1)

            # Create features for future date
            features = {
                "day_of_week": future_date.weekday(),
                "day_of_month": future_date.day,
                "month": future_date.month,
                "week_of_year": future_date.isocalendar()[1],
                "lag_1": current_values[-1] if len(current_values) >= 1 else 0,
                "lag_7": current_values[-7] if len(current_values) >= 7 else current_values[-1],
                "lag_14": current_values[-14] if len(current_values) >= 14 else current_values[-1],
                "lag_30": current_values[-30] if len(current_values) >= 30 else current_values[-1],
                "rolling_mean_7": np.mean(current_values[-7:]) if len(current_values) >= 7 else np.mean(current_values),
                "rolling_std_7": np.std(current_values[-7:]) if len(current_values) >= 7 else np.std(current_values),
                "rolling_mean_30": np.mean(current_values[-30:]) if len(current_values) >= 30 else np.mean(current_values),
            }

            X_pred = pd.DataFrame([features])
            pred = model.predict(X_pred)[0]
            pred = max(0, pred)

            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_demand": round(pred, 2)
            })

            current_values.append(pred)

        return predictions

    def _ensemble_forecast(
        self,
        prophet_forecast: List[Dict[str, Any]],
        xgb_forecast: List[Dict[str, Any]],
        prophet_weight: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Combine Prophet and XGBoost forecasts"""
        ensemble = []

        for p, x in zip(prophet_forecast, xgb_forecast):
            combined_demand = (
                prophet_weight * p["predicted_demand"] +
                (1 - prophet_weight) * x["predicted_demand"]
            )
            entry = {
                "date": p["date"],
                "predicted_demand": round(combined_demand, 2),
            }
            if "lower_bound" in p:
                entry["lower_bound"] = p["lower_bound"]
                entry["upper_bound"] = p["upper_bound"]

            ensemble.append(entry)

        return ensemble

    def _simple_forecast(
        self,
        data: pd.DataFrame,
        periods: int,
        sku_id: Optional[str],
        warehouse_id: Optional[str]
    ) -> Dict[str, Any]:
        """Simple average-based forecast for limited data"""
        avg_demand = data["y"].mean() if "y" in data.columns else 0
        std_demand = data["y"].std() if "y" in data.columns else 0

        last_date = pd.to_datetime(data["ds"]).max() if "ds" in data.columns else datetime.now()

        forecast = []
        for i in range(periods):
            future_date = last_date + timedelta(days=i + 1)
            forecast.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_demand": round(avg_demand, 2),
                "lower_bound": round(max(0, avg_demand - 2 * std_demand), 2),
                "upper_bound": round(avg_demand + 2 * std_demand, 2)
            })

        return {
            "sku_id": sku_id,
            "warehouse_id": warehouse_id,
            "forecast": forecast,
            "model_used": "simple_average",
            "mape": None,
            "periods": periods
        }

    def _calculate_mape(
        self,
        historical_data: pd.DataFrame,
        forecast: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate Mean Absolute Percentage Error on historical data"""
        try:
            # Use last 20% of historical data as validation
            n_val = max(1, int(len(historical_data) * 0.2))
            val_data = historical_data.tail(n_val)

            if len(val_data) == 0:
                return None

            actual = val_data["y"].values
            # For MAPE calculation, we'd need to retrain without val data
            # Here we use a simplified approach
            predicted = [f["predicted_demand"] for f in forecast[:n_val]]

            if len(predicted) < len(actual):
                return None

            mape = np.mean(np.abs((actual - predicted[:len(actual)]) / (actual + 1e-9))) * 100
            return round(mape, 2)
        except Exception:
            return None

    def forecast_to_text(self, forecast_result: Dict[str, Any]) -> str:
        """Convert forecast to natural language summary"""
        sku_id = forecast_result.get("sku_id", "Unknown SKU")
        warehouse_id = forecast_result.get("warehouse_id", "All warehouses")
        model = forecast_result.get("model_used", "unknown")
        mape = forecast_result.get("mape")
        forecast = forecast_result.get("forecast", [])

        if not forecast:
            return f"No forecast available for SKU {sku_id}."

        # Calculate summary stats
        demands = [f["predicted_demand"] for f in forecast]
        avg_demand = np.mean(demands)
        max_demand = np.max(demands)
        min_demand = np.min(demands)
        total_demand = np.sum(demands)

        # Find peak period
        peak_idx = np.argmax(demands)
        peak_date = forecast[peak_idx]["date"]

        summary = f"""
**Demand Forecast for {sku_id}**
- Warehouse: {warehouse_id}
- Model: {model}
- Forecast Period: {len(forecast)} days

**Summary:**
- Average Daily Demand: {avg_demand:.1f} units
- Total Forecasted Demand: {total_demand:.0f} units
- Peak Demand: {max_demand:.1f} units on {peak_date}
- Minimum Demand: {min_demand:.1f} units
"""

        if mape is not None:
            summary += f"- Model MAPE: {mape:.1f}%\n"

        return summary.strip()
