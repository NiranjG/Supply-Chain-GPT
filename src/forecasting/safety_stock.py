"""
Safety stock and reorder point calculations
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class SafetyStockCalculator:
    """
    Calculate safety stock and reorder points using various methods:
    - Statistical (standard deviation based)
    - Service level based
    - Lead time variability
    """

    def __init__(self, default_service_level: float = 0.95):
        """
        Initialize calculator

        Args:
            default_service_level: Default service level (e.g., 0.95 for 95%)
        """
        self.default_service_level = default_service_level

        # Z-scores for common service levels
        self.z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.97: 1.88,
            0.98: 2.05,
            0.99: 2.33,
            0.995: 2.58
        }

    def calculate(
        self,
        demand_data: pd.DataFrame,
        lead_time_days: float,
        lead_time_std: Optional[float] = None,
        service_level: Optional[float] = None,
        method: str = "statistical"
    ) -> Dict[str, Any]:
        """
        Calculate safety stock and reorder point

        Args:
            demand_data: DataFrame with 'y' column for daily demand
            lead_time_days: Average lead time in days
            lead_time_std: Standard deviation of lead time (optional)
            service_level: Target service level (default 0.95)
            method: Calculation method ('statistical', 'service_level', 'combined')

        Returns:
            Dictionary with safety stock, reorder point, and recommendations
        """
        service_level = service_level or self.default_service_level
        z_score = self._get_z_score(service_level)

        # Calculate demand statistics
        demand_stats = self._calculate_demand_stats(demand_data)

        if method == "statistical":
            result = self._statistical_method(
                demand_stats, lead_time_days, z_score
            )
        elif method == "service_level":
            result = self._service_level_method(
                demand_stats, lead_time_days, lead_time_std, z_score
            )
        else:  # combined
            result = self._combined_method(
                demand_stats, lead_time_days, lead_time_std, z_score
            )

        # Add metadata
        result.update({
            "service_level": service_level,
            "z_score": z_score,
            "lead_time_days": lead_time_days,
            "demand_stats": demand_stats,
            "method": method
        })

        # Generate recommendations
        result["recommendations"] = self._generate_recommendations(result)

        return result

    def _calculate_demand_stats(self, demand_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate demand statistics"""
        if "y" not in demand_data.columns:
            raise ValueError("demand_data must have 'y' column")

        demand = demand_data["y"].dropna()

        return {
            "mean_daily_demand": float(demand.mean()),
            "std_daily_demand": float(demand.std()),
            "max_daily_demand": float(demand.max()),
            "min_daily_demand": float(demand.min()),
            "cv": float(demand.std() / demand.mean()) if demand.mean() > 0 else 0,
            "sample_size": len(demand)
        }

    def _statistical_method(
        self,
        demand_stats: Dict[str, float],
        lead_time_days: float,
        z_score: float
    ) -> Dict[str, Any]:
        """
        Standard statistical safety stock formula:
        Safety Stock = Z * σd * √L

        Where:
        - Z = Z-score for service level
        - σd = Standard deviation of daily demand
        - L = Lead time in days
        """
        std_demand = demand_stats["std_daily_demand"]
        mean_demand = demand_stats["mean_daily_demand"]

        safety_stock = z_score * std_demand * np.sqrt(lead_time_days)
        reorder_point = (mean_demand * lead_time_days) + safety_stock

        return {
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "average_inventory": round(safety_stock + (mean_demand * lead_time_days) / 2, 2)
        }

    def _service_level_method(
        self,
        demand_stats: Dict[str, float],
        lead_time_days: float,
        lead_time_std: Optional[float],
        z_score: float
    ) -> Dict[str, Any]:
        """
        Service level method accounting for both demand and lead time variability:
        Safety Stock = Z * √(L * σd² + d² * σL²)

        Where:
        - L = Average lead time
        - σd = Std dev of demand
        - d = Average demand
        - σL = Std dev of lead time
        """
        std_demand = demand_stats["std_daily_demand"]
        mean_demand = demand_stats["mean_daily_demand"]
        lead_time_std = lead_time_std or 0

        # Combined variability
        combined_variance = (
            lead_time_days * (std_demand ** 2) +
            (mean_demand ** 2) * (lead_time_std ** 2)
        )

        safety_stock = z_score * np.sqrt(combined_variance)
        reorder_point = (mean_demand * lead_time_days) + safety_stock

        return {
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "average_inventory": round(safety_stock + (mean_demand * lead_time_days) / 2, 2),
            "demand_variability_component": round(z_score * std_demand * np.sqrt(lead_time_days), 2),
            "lead_time_variability_component": round(z_score * mean_demand * lead_time_std, 2) if lead_time_std else 0
        }

    def _combined_method(
        self,
        demand_stats: Dict[str, float],
        lead_time_days: float,
        lead_time_std: Optional[float],
        z_score: float
    ) -> Dict[str, Any]:
        """
        Combined method with additional buffer for demand spikes
        """
        base_result = self._service_level_method(
            demand_stats, lead_time_days, lead_time_std, z_score
        )

        # Add buffer for high CV (coefficient of variation)
        cv = demand_stats["cv"]
        if cv > 0.5:  # High variability
            buffer = base_result["safety_stock"] * 0.2
            base_result["safety_stock"] = round(base_result["safety_stock"] + buffer, 2)
            base_result["reorder_point"] = round(base_result["reorder_point"] + buffer, 2)
            base_result["high_variability_buffer"] = round(buffer, 2)

        return base_result

    def _get_z_score(self, service_level: float) -> float:
        """Get Z-score for service level"""
        if service_level in self.z_scores:
            return self.z_scores[service_level]

        # Calculate using scipy
        return float(stats.norm.ppf(service_level))

    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        demand_stats = result.get("demand_stats", {})

        # High variability warning
        cv = demand_stats.get("cv", 0)
        if cv > 0.7:
            recommendations.append(
                f"High demand variability (CV={cv:.2f}). Consider increasing safety stock "
                "or implementing demand smoothing strategies."
            )
        elif cv > 0.5:
            recommendations.append(
                f"Moderate demand variability (CV={cv:.2f}). Monitor closely for stockouts."
            )

        # Lead time recommendations
        lead_time = result.get("lead_time_days", 0)
        if lead_time > 14:
            recommendations.append(
                f"Long lead time ({lead_time} days) increases inventory holding costs. "
                "Consider alternative suppliers or local sourcing."
            )

        # Service level recommendations
        service_level = result.get("service_level", 0.95)
        safety_stock = result.get("safety_stock", 0)

        if service_level > 0.98:
            recommendations.append(
                f"Very high service level ({service_level:.0%}) requires {safety_stock:.0f} units of safety stock. "
                "Evaluate if this level is necessary for all SKUs."
            )

        # Reorder point action
        reorder_point = result.get("reorder_point", 0)
        recommendations.append(
            f"Set reorder point alert at {reorder_point:.0f} units to maintain "
            f"{service_level:.0%} service level."
        )

        return recommendations

    def calculate_economic_order_quantity(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_per_unit: float
    ) -> Dict[str, float]:
        """
        Calculate Economic Order Quantity (EOQ)

        Args:
            annual_demand: Total annual demand
            ordering_cost: Cost per order
            holding_cost_per_unit: Annual holding cost per unit

        Returns:
            Dictionary with EOQ and related metrics
        """
        if holding_cost_per_unit <= 0 or ordering_cost <= 0:
            raise ValueError("Costs must be positive")

        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        orders_per_year = annual_demand / eoq
        time_between_orders = 365 / orders_per_year

        total_ordering_cost = orders_per_year * ordering_cost
        total_holding_cost = (eoq / 2) * holding_cost_per_unit
        total_cost = total_ordering_cost + total_holding_cost

        return {
            "eoq": round(eoq, 0),
            "orders_per_year": round(orders_per_year, 1),
            "time_between_orders_days": round(time_between_orders, 1),
            "total_annual_ordering_cost": round(total_ordering_cost, 2),
            "total_annual_holding_cost": round(total_holding_cost, 2),
            "total_annual_cost": round(total_cost, 2)
        }

    def to_text_summary(self, result: Dict[str, Any], sku_id: str = "Unknown") -> str:
        """Convert calculation results to natural language summary"""
        safety_stock = result.get("safety_stock", 0)
        reorder_point = result.get("reorder_point", 0)
        service_level = result.get("service_level", 0.95)
        recommendations = result.get("recommendations", [])

        summary = f"""
**Safety Stock Analysis for SKU: {sku_id}**

**Calculated Values:**
- Safety Stock: {safety_stock:.0f} units
- Reorder Point: {reorder_point:.0f} units
- Target Service Level: {service_level:.0%}

**Recommendations:**
"""
        for rec in recommendations:
            summary += f"- {rec}\n"

        return summary.strip()
