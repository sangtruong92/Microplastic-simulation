import unittest

from microplastic_simulation import ExperimentSetup, simulate_microplastic


class TestMicroplasticSimulation(unittest.TestCase):
    def test_simulation_returns_expected_time_points(self):
        setup = ExperimentSetup(
            water_volume_liters=100,
            flow_rate_lph=10,
            temperature_celsius=20,
            turbulence_factor=1,
            duration_hours=2,
            time_step_hours=1,
        )

        results = simulate_microplastic(initial_concentration_mg_l=5, setup=setup)

        self.assertEqual([point["time_hours"] for point in results], [0.0, 1.0, 2.0])

    def test_concentration_drops_without_inflow(self):
        setup = ExperimentSetup(
            water_volume_liters=50,
            flow_rate_lph=0,
            temperature_celsius=20,
            turbulence_factor=1,
            duration_hours=5,
            time_step_hours=1,
            inflow_concentration_mg_l=0,
            degradation_rate_per_hour=0.02,
            settling_rate_per_hour=0.05,
        )

        results = simulate_microplastic(initial_concentration_mg_l=10, setup=setup)

        self.assertGreater(results[0]["concentration_mg_l"], results[-1]["concentration_mg_l"])

    def test_invalid_setup_raises_error(self):
        setup = ExperimentSetup(
            water_volume_liters=0,
            flow_rate_lph=5,
            temperature_celsius=20,
            turbulence_factor=1,
            duration_hours=1,
            time_step_hours=1,
        )

        with self.assertRaises(ValueError):
            simulate_microplastic(initial_concentration_mg_l=1, setup=setup)


if __name__ == "__main__":
    unittest.main()
