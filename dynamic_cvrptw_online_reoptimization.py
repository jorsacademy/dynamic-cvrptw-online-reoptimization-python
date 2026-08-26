from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    service_time: float
    arrival_time: float
    ready_time: float
    due_time: float
    is_served: bool = False
    assigned_vehicle: int = -1


@dataclass
class Vehicle:
    id: int
    capacity: float
    position: Tuple[float, float]
    current_time: float = 0.0
    delivered_load: float = 0.0
    planned_route: List[int] = field(default_factory=list)
    total_distance: float = 0.0
    target_customer: int = -1
    phase: str = "idle"  # idle, moving, waiting, service
    phase_remaining: float = 0.0


class DynamicCVRPTWSolver:
    """
    Educational dynamic CVRP with time windows and time-varying traffic zones.

    Model scope
    -----------
    - Customers are revealed online at `arrival_time`.
    - Each vehicle performs one capacity-limited route over the planning horizon.
    - Assigned-but-unserved demand reserves vehicle capacity.
    - Served demand remains consumed; capacity is not restored unless a depot
      replenishment policy is explicitly modeled (this implementation does not).
    - A moving/waiting/servicing target is committed during reoptimization.
    - Future unserved stops may be reassigned.
    - Time-window feasibility uses service-start <= due_time.
    - Planned routes must be able to return to the depot within the horizon.

    Routing policy
    --------------
    New customers are inserted by minimum *marginal* route distance among
    feasible insertion positions. Traffic affects travel-time feasibility.
    Traffic updates trigger a route reoptimization of non-committed stops.

    This is a heuristic online simulator, not an exact DVRP optimizer.
    """

    EPS = 1e-10

    def __init__(
        self,
        depot: Tuple[float, float],
        num_vehicles: int,
        vehicle_capacity: float,
        planning_horizon: float,
    ):
        if num_vehicles <= 0:
            raise ValueError("num_vehicles must be positive")
        if vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive")
        if planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive")

        self.depot = tuple(map(float, depot))
        self.num_vehicles = int(num_vehicles)
        self.vehicle_capacity = float(vehicle_capacity)
        self.planning_horizon = float(planning_horizon)

        self.vehicles = [
            Vehicle(i, self.vehicle_capacity, self.depot)
            for i in range(self.num_vehicles)
        ]

        self.customers: Dict[int, Customer] = {}
        self.revealed_customers: Set[int] = set()
        self.served_customers: Set[int] = set()
        self.rejected_customers: Set[int] = set()

        self.current_time = 0.0
        self.event_queue: List[Tuple[float, str, int]] = []
        self.traffic_zones: Dict[str, Tuple[Tuple[float, float], float, float]] = {}
        self.reoptimization_triggered = False

    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def get_traffic_factor(self, position: Tuple[float, float]) -> float:
        factor = 1.0
        for center, radius, zone_factor in self.traffic_zones.values():
            if self.distance(position, center) <= radius + self.EPS:
                factor *= zone_factor
        return factor

    def add_customer(self, customer: Customer) -> None:
        if customer.id <= 0:
            raise ValueError("customer id must be positive")
        if customer.id in self.customers:
            raise ValueError(f"duplicate customer id: {customer.id}")
        if customer.demand <= 0 or customer.demand > self.vehicle_capacity + self.EPS:
            raise ValueError("customer demand must be in (0, vehicle_capacity]")
        if customer.service_time < 0:
            raise ValueError("service_time must be nonnegative")
        if not (0 <= customer.arrival_time <= self.planning_horizon):
            raise ValueError("arrival_time outside planning horizon")
        if customer.ready_time > customer.due_time:
            raise ValueError("ready_time must not exceed due_time")

        self.customers[customer.id] = customer
        heapq.heappush(
            self.event_queue,
            (float(customer.arrival_time), "arrival", customer.id),
        )

    def update_traffic(
        self,
        zone_name: str,
        center: Tuple[float, float],
        radius: float,
        speed_factor: float,
    ) -> None:
        if radius <= 0 or speed_factor <= 0:
            raise ValueError("radius and speed_factor must be positive")
        self.traffic_zones[zone_name] = (
            tuple(map(float, center)),
            float(radius),
            float(speed_factor),
        )
        self.reoptimization_triggered = True

    def _reserved_load(self, vehicle: Vehicle) -> float:
        return sum(
            self.customers[cid].demand
            for cid in vehicle.planned_route
            if cid not in self.served_customers
        )

    def remaining_capacity(self, vehicle: Vehicle) -> float:
        return (
            vehicle.capacity
            - vehicle.delivered_load
            - self._reserved_load(vehicle)
        )

    def _locked_prefix_length(self, vehicle: Vehicle) -> int:
        if (
            vehicle.target_customer != -1
            and vehicle.phase in {"moving", "waiting", "service"}
            and vehicle.planned_route
            and vehicle.planned_route[0] == vehicle.target_customer
        ):
            return 1
        return 0

    def _anchor_after_locked_target(
        self, vehicle: Vehicle
    ) -> Tuple[Tuple[float, float], float]:
        if self._locked_prefix_length(vehicle) == 0:
            return vehicle.position, max(vehicle.current_time, self.current_time)

        customer = self.customers[vehicle.target_customer]

        if vehicle.phase == "moving":
            factor = self.get_traffic_factor(vehicle.position)
            travel = self.distance(vehicle.position, (customer.x, customer.y)) / factor
            arrival = vehicle.current_time + travel
            start = max(arrival, customer.ready_time)
            finish = start + customer.service_time
        elif vehicle.phase == "waiting":
            finish = vehicle.current_time + vehicle.phase_remaining + customer.service_time
        else:
            finish = vehicle.current_time + vehicle.phase_remaining

        return (customer.x, customer.y), finish

    def _tail_route_cost_and_feasibility(
        self,
        vehicle: Vehicle,
        tail: List[int],
    ) -> Tuple[float, bool]:
        pos, t = self._anchor_after_locked_target(vehicle)
        total_distance = 0.0

        for cid in tail:
            c = self.customers[cid]
            dist = self.distance(pos, (c.x, c.y))
            factor = self.get_traffic_factor(pos)
            travel_time = dist / factor
            arrival = t + travel_time
            start_service = max(arrival, c.ready_time)

            if start_service > c.due_time + self.EPS:
                return math.inf, False

            total_distance += dist
            t = start_service + c.service_time
            pos = (c.x, c.y)

        return_distance = self.distance(pos, self.depot)
        return_time = return_distance / self.get_traffic_factor(pos)
        if t + return_time > self.planning_horizon + self.EPS:
            return math.inf, False

        total_distance += return_distance
        return total_distance, True

    def calculate_insertion_cost(
        self,
        vehicle: Vehicle,
        customer_id: int,
        position: int,
    ) -> Tuple[float, bool]:
        customer = self.customers[customer_id]
        if self.remaining_capacity(vehicle) + self.EPS < customer.demand:
            return math.inf, False

        prefix = self._locked_prefix_length(vehicle)
        if position < prefix or position > len(vehicle.planned_route):
            return math.inf, False

        base_tail = vehicle.planned_route[prefix:]
        base_cost, base_feasible = self._tail_route_cost_and_feasibility(
            vehicle, base_tail
        )
        if not base_feasible:
            return math.inf, False

        relative_position = position - prefix
        candidate_tail = base_tail[:]
        candidate_tail.insert(relative_position, customer_id)

        candidate_cost, feasible = self._tail_route_cost_and_feasibility(
            vehicle, candidate_tail
        )
        if not feasible:
            return math.inf, False

        return candidate_cost - base_cost, True

    def find_best_insertion(self, customer_id: int) -> Tuple[int, int, float]:
        best = (-1, -1, math.inf)

        for vehicle in self.vehicles:
            prefix = self._locked_prefix_length(vehicle)
            for pos in range(prefix, len(vehicle.planned_route) + 1):
                marginal, feasible = self.calculate_insertion_cost(
                    vehicle, customer_id, pos
                )
                candidate = (vehicle.id, pos, marginal)
                if feasible and (
                    marginal < best[2] - self.EPS
                    or (
                        math.isclose(marginal, best[2], abs_tol=self.EPS)
                        and (vehicle.id, pos) < (best[0], best[1])
                    )
                ):
                    best = candidate

        return best

    def assign_customer_to_vehicle(
        self,
        customer_id: int,
        vehicle_id: int,
        position: int,
    ) -> None:
        customer = self.customers[customer_id]
        vehicle = self.vehicles[vehicle_id]

        if customer.is_served or customer_id in self.rejected_customers:
            raise ValueError("cannot assign served/rejected customer")
        if customer.assigned_vehicle != -1:
            raise ValueError("customer is already assigned")
        if self.remaining_capacity(vehicle) + self.EPS < customer.demand:
            raise ValueError("insufficient vehicle capacity")

        vehicle.planned_route.insert(position, customer_id)
        customer.assigned_vehicle = vehicle_id

    def handle_customer_arrival(self, customer_id: int) -> None:
        self.revealed_customers.add(customer_id)
        vehicle_id, position, _ = self.find_best_insertion(customer_id)

        if vehicle_id == -1:
            self.rejected_customers.add(customer_id)
            self.customers[customer_id].assigned_vehicle = -1
            return

        self.assign_customer_to_vehicle(customer_id, vehicle_id, position)

    def reoptimize_all_routes(self) -> None:
        flexible: List[int] = []

        for vehicle in self.vehicles:
            prefix = self._locked_prefix_length(vehicle)
            locked = vehicle.planned_route[:prefix]
            flexible.extend(vehicle.planned_route[prefix:])

            for cid in vehicle.planned_route[prefix:]:
                self.customers[cid].assigned_vehicle = -1

            vehicle.planned_route = locked

        flexible = sorted(
            set(flexible),
            key=lambda cid: (
                self.customers[cid].due_time,
                self.customers[cid].arrival_time,
                cid,
            ),
        )

        for cid in flexible:
            if cid in self.served_customers or cid in self.rejected_customers:
                continue
            vehicle_id, position, _ = self.find_best_insertion(cid)
            if vehicle_id == -1:
                self.rejected_customers.add(cid)
                self.customers[cid].assigned_vehicle = -1
            else:
                self.assign_customer_to_vehicle(cid, vehicle_id, position)

        self.reoptimization_triggered = False

    def _start_next_target_if_needed(self, vehicle: Vehicle) -> None:
        if vehicle.phase != "idle" or not vehicle.planned_route:
            return
        cid = vehicle.planned_route[0]
        vehicle.target_customer = cid
        vehicle.phase = "moving"
        vehicle.phase_remaining = 0.0

    def _complete_service(self, vehicle: Vehicle) -> None:
        cid = vehicle.target_customer
        customer = self.customers[cid]

        customer.is_served = True
        self.served_customers.add(cid)
        vehicle.delivered_load += customer.demand

        if not vehicle.planned_route or vehicle.planned_route[0] != cid:
            raise RuntimeError("committed target missing from planned route")
        vehicle.planned_route.pop(0)

        vehicle.target_customer = -1
        vehicle.phase = "idle"
        vehicle.phase_remaining = 0.0

    def _advance_vehicle(self, vehicle: Vehicle, duration: float) -> None:
        remaining = float(duration)

        while remaining > self.EPS:
            self._start_next_target_if_needed(vehicle)

            if vehicle.phase == "idle":
                vehicle.current_time += remaining
                break

            customer = self.customers[vehicle.target_customer]

            if vehicle.phase == "moving":
                target = (customer.x, customer.y)
                dist = self.distance(vehicle.position, target)
                factor = self.get_traffic_factor(vehicle.position)
                travel_time = dist / factor

                if travel_time > remaining + self.EPS:
                    move_distance = factor * remaining
                    ratio = move_distance / dist if dist > self.EPS else 1.0
                    vehicle.position = (
                        vehicle.position[0]
                        + (target[0] - vehicle.position[0]) * ratio,
                        vehicle.position[1]
                        + (target[1] - vehicle.position[1]) * ratio,
                    )
                    vehicle.total_distance += move_distance
                    vehicle.current_time += remaining
                    remaining = 0.0
                else:
                    vehicle.position = target
                    vehicle.total_distance += dist
                    vehicle.current_time += travel_time
                    remaining -= travel_time

                    wait = max(0.0, customer.ready_time - vehicle.current_time)
                    if wait > self.EPS:
                        vehicle.phase = "waiting"
                        vehicle.phase_remaining = wait
                    else:
                        if vehicle.current_time > customer.due_time + self.EPS:
                            raise RuntimeError("committed route violated time window")
                        vehicle.phase = "service"
                        vehicle.phase_remaining = customer.service_time

            elif vehicle.phase == "waiting":
                step = min(remaining, vehicle.phase_remaining)
                vehicle.phase_remaining -= step
                vehicle.current_time += step
                remaining -= step
                if vehicle.phase_remaining <= self.EPS:
                    if vehicle.current_time > customer.due_time + self.EPS:
                        raise RuntimeError("committed route violated time window")
                    vehicle.phase = "service"
                    vehicle.phase_remaining = customer.service_time

            elif vehicle.phase == "service":
                step = min(remaining, vehicle.phase_remaining)
                vehicle.phase_remaining -= step
                vehicle.current_time += step
                remaining -= step
                if vehicle.phase_remaining <= self.EPS:
                    self._complete_service(vehicle)

    def update_vehicles(self, time_delta: float) -> None:
        if time_delta < -self.EPS:
            raise ValueError("time_delta must be nonnegative")
        for vehicle in self.vehicles:
            self._advance_vehicle(vehicle, max(0.0, time_delta))

    def simulate_step(self, time_delta: float) -> None:
        if time_delta <= 0:
            raise ValueError("time_delta must be positive")

        target_time = min(
            self.current_time + time_delta,
            self.planning_horizon,
        )

        while self.event_queue and self.event_queue[0][0] <= target_time + self.EPS:
            event_time, event_type, customer_id = heapq.heappop(self.event_queue)

            advance = event_time - self.current_time
            if advance > self.EPS:
                self.update_vehicles(advance)
                self.current_time = event_time

            if event_type == "arrival":
                self.handle_customer_arrival(customer_id)

        remaining = target_time - self.current_time
        if remaining > self.EPS:
            self.update_vehicles(remaining)
            self.current_time = target_time

        if self.reoptimization_triggered:
            self.reoptimize_all_routes()

        for vehicle in self.vehicles:
            if not math.isclose(
                vehicle.current_time,
                self.current_time,
                abs_tol=1e-8,
            ):
                raise RuntimeError("vehicle/global clocks are inconsistent")

    def get_status(self) -> Dict:
        return {
            "time": round(self.current_time, 2),
            "served": len(self.served_customers),
            "revealed": len(self.revealed_customers),
            "total": len(self.customers),
            "rejected": len(self.rejected_customers),
            "total_distance": round(
                sum(v.total_distance for v in self.vehicles), 2
            ),
            "traffic_zones": len(self.traffic_zones),
            "vehicles": [
                {
                    "id": v.id,
                    "pos": (
                        round(v.position[0], 2),
                        round(v.position[1], 2),
                    ),
                    "route": list(v.planned_route),
                    "remaining_capacity": round(
                        self.remaining_capacity(v), 2
                    ),
                    "delivered_load": round(v.delivered_load, 2),
                    "distance": round(v.total_distance, 2),
                    "phase": v.phase,
                    "target": v.target_customer,
                }
                for v in self.vehicles
            ],
        }


def create_reference_solver() -> DynamicCVRPTWSolver:
    solver = DynamicCVRPTWSolver(
        depot=(50, 50),
        num_vehicles=4,
        vehicle_capacity=100,
        planning_horizon=220,
    )

    data = [
        (1, 60, 70, 20, 3, 0, 0, 100),
        (2, 40, 60, 15, 3, 0, 0, 100),
        (3, 70, 40, 25, 3, 0, 0, 120),
        (4, 30, 30, 18, 3, 0, 0, 120),
        (5, 80, 80, 22, 3, 10, 10, 140),
        (6, 20, 70, 12, 3, 15, 15, 150),
        (7, 65, 25, 28, 3, 20, 20, 160),
        (8, 45, 85, 16, 3, 25, 25, 170),
        (9, 85, 55, 19, 3, 30, 30, 180),
        (10, 35, 45, 21, 3, 35, 35, 190),
        (11, 55, 35, 17, 3, 40, 40, 200),
        (12, 75, 65, 23, 3, 45, 45, 210),
    ]

    for row in data:
        solver.add_customer(Customer(*row))

    return solver


if __name__ == "__main__":
    solver = create_reference_solver()

    for t in range(150):
        solver.simulate_step(1.0)

        if t == 20:
            solver.update_traffic(
                "downtown",
                (60, 60),
                radius=20,
                speed_factor=0.5,
            )

        if t == 50:
            solver.update_traffic(
                "highway",
                (70, 40),
                radius=15,
                speed_factor=1.5,
            )

    print(solver.get_status())
