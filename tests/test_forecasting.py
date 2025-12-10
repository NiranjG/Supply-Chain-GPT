"""
Tests for forecasting module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.forecasting.demand_forecaster import DemandForecaster
from src.forecasting.safety_stock import SafetyStockCalculator


class TestDemandForecaster:
    """Tests for DemandForecaster class"""

    @pytest.fixture
    def forecaster(self):
        """Create forecaster instance"""
        return DemandForecaster(use_prophet=False, use_xgboost=True)

    @pytest.fixture
    def historical_data(self):
        """Generate historical demand data"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        base_demand = 100
        # Add seasonality and noise
        demand = base_demand + 20 * np.sin(np.arange(90) * 2 * np.pi / 7) + np.random.randn(90) * 10
        demand = np.maximum(demand, 0)

        return pd.DataFrame({
            'ds': dates,
            'y': demand
        })

    @pytest.fixture
    def minimal_data(self):
        """Minimal data for fallback testing"""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        return pd.DataFrame({
            'ds': dates,
            'y': [100, 110, 95, 105, 100]
        })

    def test_forecast_returns_predictions(self, forecaster, historical_data):
        """Test that forecast returns predictions"""
        result = forecaster.forecast(
            historical_data=historical_data,
            periods=7,
            sku_id="SKU-001"
        )

        assert "forecast" in result
        assert len(result["forecast"]) == 7
        assert all("date" in f for f in result["forecast"])
        assert all("predicted_demand" in f for f in result["forecast"])

    def test_forecast_predictions_are_positive(self, forecaster, historical_data):
        """Test that predictions are non-negative"""
        result = forecaster.forecast(historical_data, periods=30)

        for prediction in result["forecast"]:
            assert prediction["predicted_demand"] >= 0

    def test_forecast_with_minimal_data(self, forecaster, minimal_data):
        """Test forecast fallback with minimal data"""
        result = forecaster.forecast(minimal_data, periods=7)

        assert "forecast" in result
        assert result["model_used"] == "simple_average"

    def test_forecast_metadata(self, forecaster, historical_data):
        """Test that forecast includes metadata"""
        result = forecaster.forecast(
            historical_data,
            periods=14,
            sku_id="SKU-TEST",
            warehouse_id="WH-001"
        )

        assert result["sku_id"] == "SKU-TEST"
        assert result["warehouse_id"] == "WH-001"
        assert result["periods"] == 14
        assert "model_used" in result

    def test_forecast_to_text(self, forecaster, historical_data):
        """Test text summary generation"""
        result = forecaster.forecast(historical_data, periods=7, sku_id="SKU-001")
        summary = forecaster.forecast_to_text(result)

        assert "SKU-001" in summary
        assert "Demand" in summary or "demand" in summary
        assert "units" in summary.lower()

    def test_empty_data_handling(self, forecaster):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame({'ds': [], 'y': []})
        result = forecaster.forecast(empty_df, periods=7)

        assert "forecast" in result


class TestXGBoostForecaster:
    """Tests specifically for XGBoost forecasting"""

    @pytest.fixture
    def forecaster(self):
        """Create XGBoost-only forecaster"""
        return DemandForecaster(use_prophet=False, use_xgboost=True)

    @pytest.fixture
    def seasonal_data(self):
        """Generate data with clear weekly seasonality"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')

        # Weekly pattern: higher on weekdays, lower on weekends
        demand = []
        for i, date in enumerate(dates):
            base = 100
            # Day of week effect
            if date.weekday() < 5:  # Weekday
                base += 30
            else:  # Weekend
                base -= 20
            # Add noise
            demand.append(base + np.random.randn() * 5)

        return pd.DataFrame({
            'ds': dates,
            'y': demand
        })

    def test_xgboost_captures_patterns(self, forecaster, seasonal_data):
        """Test that XGBoost can capture demand patterns"""
        result = forecaster.forecast(seasonal_data, periods=14)

        # Should use xgboost model
        assert result["model_used"] in ["xgboost", "ensemble"]
        assert len(result["forecast"]) == 14


class TestSafetyStockCalculator:
    """Tests for SafetyStockCalculator class"""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return SafetyStockCalculator(default_service_level=0.95)

    @pytest.fixture
    def demand_data(self):
        """Generate demand data"""
        np.random.seed(42)
        return pd.DataFrame({
            'y': 100 + np.random.randn(90) * 20
        })

    @pytest.fixture
    def high_variability_data(self):
        """Generate high variability demand data"""
        np.random.seed(42)
        return pd.DataFrame({
            'y': 100 + np.random.randn(90) * 60  # High std dev
        })

    def test_calculate_safety_stock(self, calculator, demand_data):
        """Test basic safety stock calculation"""
        result = calculator.calculate(
            demand_data=demand_data,
            lead_time_days=7
        )

        assert "safety_stock" in result
        assert "reorder_point" in result
        assert result["safety_stock"] > 0
        assert result["reorder_point"] > result["safety_stock"]

    def test_service_level_affects_safety_stock(self, calculator, demand_data):
        """Test that higher service level increases safety stock"""
        result_95 = calculator.calculate(demand_data, lead_time_days=7, service_level=0.95)
        result_99 = calculator.calculate(demand_data, lead_time_days=7, service_level=0.99)

        assert result_99["safety_stock"] > result_95["safety_stock"]

    def test_lead_time_affects_safety_stock(self, calculator, demand_data):
        """Test that longer lead time increases safety stock"""
        result_short = calculator.calculate(demand_data, lead_time_days=3)
        result_long = calculator.calculate(demand_data, lead_time_days=14)

        assert result_long["safety_stock"] > result_short["safety_stock"]

    def test_statistical_method(self, calculator, demand_data):
        """Test statistical calculation method"""
        result = calculator.calculate(
            demand_data=demand_data,
            lead_time_days=7,
            method="statistical"
        )

        assert result["method"] == "statistical"
        assert "safety_stock" in result

    def test_service_level_method(self, calculator, demand_data):
        """Test service level method with lead time variability"""
        result = calculator.calculate(
            demand_data=demand_data,
            lead_time_days=7,
            lead_time_std=2,
            method="service_level"
        )

        assert "demand_variability_component" in result
        assert "lead_time_variability_component" in result

    def test_recommendations_generated(self, calculator, high_variability_data):
        """Test that recommendations are generated"""
        result = calculator.calculate(
            demand_data=high_variability_data,
            lead_time_days=7
        )

        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_high_variability_warning(self, calculator, high_variability_data):
        """Test that high variability generates warning"""
        result = calculator.calculate(high_variability_data, lead_time_days=7)

        # Should have a recommendation about variability
        recommendations = " ".join(result["recommendations"])
        assert "variability" in recommendations.lower()

    def test_eoq_calculation(self, calculator):
        """Test Economic Order Quantity calculation"""
        result = calculator.calculate_economic_order_quantity(
            annual_demand=10000,
            ordering_cost=50,
            holding_cost_per_unit=2
        )

        assert "eoq" in result
        assert result["eoq"] > 0
        assert "orders_per_year" in result
        assert "total_annual_cost" in result

    def test_eoq_with_zero_cost_raises(self, calculator):
        """Test that zero costs raise error"""
        with pytest.raises(ValueError):
            calculator.calculate_economic_order_quantity(
                annual_demand=10000,
                ordering_cost=0,
                holding_cost_per_unit=2
            )

    def test_to_text_summary(self, calculator, demand_data):
        """Test text summary generation"""
        result = calculator.calculate(demand_data, lead_time_days=7)
        summary = calculator.to_text_summary(result, sku_id="SKU-TEST")

        assert "SKU-TEST" in summary
        assert "Safety Stock" in summary
        assert "Reorder Point" in summary

    def test_demand_stats_calculation(self, calculator, demand_data):
        """Test demand statistics calculation"""
        stats = calculator._calculate_demand_stats(demand_data)

        assert "mean_daily_demand" in stats
        assert "std_daily_demand" in stats
        assert "cv" in stats
        assert stats["mean_daily_demand"] > 0
