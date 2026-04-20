# Microplastic-simulation

Simulation for microplastic behavior in water.

## Example

```python
from microplastic_simulation import ExperimentSetup, simulate_microplastic

setup = ExperimentSetup(
    water_volume_liters=100,
    flow_rate_lph=10,
    temperature_celsius=22,
    turbulence_factor=1.2,
    duration_hours=8,
    time_step_hours=0.5,
    inflow_concentration_mg_l=0.3,
)

results = simulate_microplastic(initial_concentration_mg_l=2.0, setup=setup)
print(results[:3])
```

Each row in `results` contains:
- `time_hours`
- `concentration_mg_l`
- `total_mass_mg`
