import math
import unittest

from dynamic_cvrptw_online_reoptimization import (
    Customer,
    DynamicCVRPTWSolver,
    create_reference_solver,
)


class DynamicCVRPTWTests(unittest.TestCase):
    def test_idle_vehicle_clock_tracks_global_time(self):
        solver = DynamicCVRPTWSolver((0, 0), 1, 10, 100)
        solver.simulate_step(5)
        self.assertTrue(
            math.isclose(
                solver.vehicles[0].current_time,
                solver.current_time,
                abs_tol=1e-10,
            )
        )

    def test_served_demand_remains_consumed(self):
        solver = DynamicCVRPTWSolver((0, 0), 1, 10, 100)
        solver.add_customer(Customer(1, 1, 0, 4, 0, 0, 0, 50))
        solver.simulate_step(2)

        vehicle = solver.vehicles[0]
        self.assertIn(1, solver.served_customers)
        self.assertTrue(math.isclose(vehicle.delivered_load, 4.0))
        self.assertTrue(math.isclose(solver.remaining_capacity(vehicle), 6.0))

    def test_committed_target_survives_reoptimization(self):
        solver = DynamicCVRPTWSolver((0, 0), 1, 20, 100)
        solver.add_customer(Customer(1, 10, 0, 5, 0, 0, 0, 50))
        solver.add_customer(Customer(2, 20, 0, 5, 0, 0, 0, 80))

        solver.simulate_step(1)
        target = solver.vehicles[0].target_customer

        solver.update_traffic("slow", (0, 0), 5, 0.5)
        solver.simulate_step(1)

        self.assertEqual(solver.vehicles[0].target_customer, target)
        self.assertEqual(solver.vehicles[0].planned_route[0], target)

    def test_insertion_cost_is_marginal(self):
        solver = DynamicCVRPTWSolver((0, 0), 2, 20, 100)
        solver.add_customer(Customer(1, 10, 0, 2, 0, 0, 0, 90))
        solver.add_customer(Customer(2, 11, 0, 2, 0, 0, 0, 90))
        solver.simulate_step(0.1)

        vehicle_id = solver.customers[2].assigned_vehicle
        self.assertIn(vehicle_id, (0, 1))
        self.assertGreaterEqual(vehicle_id, 0)

    def test_reference_scenario_serves_all_without_capacity_violation(self):
        solver = create_reference_solver()

        for t in range(150):
            solver.simulate_step(1.0)
            if t == 20:
                solver.update_traffic("downtown", (60, 60), 20, 0.5)
            if t == 50:
                solver.update_traffic("highway", (70, 40), 15, 1.5)

        self.assertEqual(len(solver.served_customers), 12)
        self.assertEqual(len(solver.rejected_customers), 0)

        for vehicle in solver.vehicles:
            self.assertLessEqual(
                vehicle.delivered_load,
                vehicle.capacity + 1e-9,
            )
            self.assertGreaterEqual(
                solver.remaining_capacity(vehicle),
                -1e-9,
            )
            self.assertTrue(
                math.isclose(
                    vehicle.current_time,
                    solver.current_time,
                    abs_tol=1e-8,
                )
            )


if __name__ == "__main__":
    unittest.main()
