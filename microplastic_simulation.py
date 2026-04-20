from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ExperimentSetup:
    """Parameters that define the experimental setup for the simulation."""

    water_volume_liters: float
    flow_rate_lph: float
    temperature_celsius: float
    turbulence_factor: float
    duration_hours: float
    time_step_hours: float
    inflow_concentration_mg_l: float = 0.0
    degradation_rate_per_hour: float = 0.005
    settling_rate_per_hour: float = 0.01


def _validate_setup(setup: ExperimentSetup, initial_concentration_mg_l: float) -> None:
    if setup.water_volume_liters <= 0:
        raise ValueError("water_volume_liters must be greater than 0")
    if setup.flow_rate_lph < 0:
        raise ValueError("flow_rate_lph must be non-negative")
    if setup.turbulence_factor < 0:
        raise ValueError("turbulence_factor must be non-negative")
    if setup.duration_hours <= 0:
        raise ValueError("duration_hours must be greater than 0")
    if setup.time_step_hours <= 0:
        raise ValueError("time_step_hours must be greater than 0")
    if setup.inflow_concentration_mg_l < 0:
        raise ValueError("inflow_concentration_mg_l must be non-negative")
    if setup.degradation_rate_per_hour < 0:
        raise ValueError("degradation_rate_per_hour must be non-negative")
    if setup.settling_rate_per_hour < 0:
        raise ValueError("settling_rate_per_hour must be non-negative")
    if initial_concentration_mg_l < 0:
        raise ValueError("initial_concentration_mg_l must be non-negative")


def simulate_microplastic(
    initial_concentration_mg_l: float,
    setup: ExperimentSetup,
) -> List[Dict[str, float]]:
    """
    Simulate microplastic concentration in water over time.

    Returns a time series where each item includes:
      - time_hours
      - concentration_mg_l
      - total_mass_mg
    """

    _validate_setup(setup, initial_concentration_mg_l)

    concentration = initial_concentration_mg_l
    time_hours = 0.0

    # Higher temperatures increase degradation, and turbulence reduces settling.
    temperature_effect = max(0.0, 1.0 + 0.02 * (setup.temperature_celsius - 20.0))
    settling_effect = setup.settling_rate_per_hour / (1.0 + setup.turbulence_factor)
    removal_rate = setup.degradation_rate_per_hour * temperature_effect + settling_effect
    washout_rate = setup.flow_rate_lph / setup.water_volume_liters
    inflow_rate = setup.inflow_concentration_mg_l * washout_rate

    results: List[Dict[str, float]] = []

    while time_hours <= setup.duration_hours + 1e-12:
        results.append(
            {
                "time_hours": round(time_hours, 10),
                "concentration_mg_l": max(0.0, concentration),
                "total_mass_mg": max(0.0, concentration) * setup.water_volume_liters,
            }
        )

        net_change = inflow_rate - (removal_rate + washout_rate) * concentration
        concentration = max(0.0, concentration + net_change * setup.time_step_hours)
        time_hours += setup.time_step_hours

    return results
