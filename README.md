# Dynamic CVRPTW with Online Reoptimization in Python

Educational dynamic Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) simulator with online customer arrivals and traffic-triggered route reoptimization.

## Problem scope

The implementation models:

- online customer revelation through `arrival_time` events;
- vehicle capacity constraints;
- customer ready/due time windows;
- service times;
- time-varying traffic zones represented by local speed factors;
- rolling insertion of newly revealed customers;
- route reoptimization after traffic updates;
- committed-target preservation for vehicles already moving, waiting, or servicing.

This is a heuristic online routing simulator. It does **not** provide an exact DVRP optimality certificate.

## State and capacity semantics

Each vehicle has two distinct load concepts:

- `delivered_load`: demand already served during the current route;
- reserved load: demand of assigned-but-unserved customers in `planned_route`.

Remaining capacity is therefore:

```text
capacity - delivered_load - reserved_load
```

Served demand remains consumed. Capacity is not restored after service because this implementation does not model depot replenishment or multi-trip reloads.

## Online insertion

Newly revealed customers are tested at every feasible insertion position. Candidate feasibility checks include:

- remaining vehicle capacity;
- current vehicle location and clock;
- committed target, if any;
- customer time windows;
- service times;
- traffic-adjusted travel times;
- ability to return to the depot before the planning horizon.

The insertion criterion uses **marginal route distance** rather than comparing absolute route totals.

## Traffic-triggered reoptimization

A traffic update marks the routing plan for reoptimization. During reoptimization:

- the customer currently being approached, waited for, or serviced remains locked;
- only future flexible stops may be reassigned;
- delivered load and current vehicle state are preserved;
- flexible stops are rebuilt deterministically using earliest due time and insertion feasibility.

This avoids teleporting or reallocating a customer after a vehicle has already committed to that stop.

## Reference scenario

The bundled scenario contains:

- 12 dynamically revealed customers;
- 4 vehicles;
- vehicle capacity 100;
- customer time windows;
- two traffic updates during simulation.

Validated reference result after 150 time units:

| Metric | Value |
|---|---:|
| Served | 12 / 12 |
| Rejected | 0 |
| Total driven distance | 283.67 |
| Traffic zones | 2 |

Final delivered loads are 58, 38, 89, and 51, all within the 100-unit vehicle capacity.

## Validation

The regression suite checks:

- idle vehicle clocks remain synchronized with global simulation time;
- served demand remains capacity-consuming;
- committed targets survive traffic reoptimization;
- insertion logic uses feasible marginal insertion;
- the reference scenario serves all 12 customers without capacity violation.

Run:

```bash
python -m unittest discover -s tests -v
```

## Usage

```python
from dynamic_cvrptw_online_reoptimization import create_reference_solver

solver = create_reference_solver()

for t in range(150):
    solver.simulate_step(1.0)

    if t == 20:
        solver.update_traffic("downtown", (60, 60), 20, 0.5)

    if t == 50:
        solver.update_traffic("highway", (70, 40), 15, 1.5)

print(solver.get_status())
```

## Requirements

- Python 3.10+
- NumPy

## Limitations

The traffic model uses position-local multiplicative speed factors rather than edge-specific travel-time functions. It does not model multi-trip replenishment, stochastic travel times, driver breaks, heterogeneous fleets, pickup-and-delivery constraints, or an exact rolling-horizon optimization model.
