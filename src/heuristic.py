"""Heuristic functions for A* search in multi-story parking navigation.

All heuristics are designed to be admissible and consistent for the parking
navigation domain, ensuring A* optimality.
"""

import math
from typing import Tuple


def manhattan_3d(current: Tuple[int, int, int], goal: Tuple[int, int, int],
                 floor_weight: float = 1.0) -> float:
    """
    3D Manhattan Distance heuristic for multi-floor navigation.

    h(n) = |x_n - x_g| + |y_n - y_g| + w * |z_n - z_g|

    where (x, y) are row/col coordinates and z is the floor index.

    Proof of Admissibility:
        - Each horizontal/vertical move on the same floor costs >= 1.0
        - Each floor transition costs >= 1.5 (ramp down, cheapest inter-floor move)
        - With floor_weight = 1.0 < 1.5, each floor difference term underestimates
        - Therefore h(n) <= h*(n) for all nodes n ✓

    Proof of Consistency (Triangle Inequality):
        - For same-floor moves: |h(n) - h(n')| <= 1 = c(n, n')
        - For inter-floor moves: |h(n) - h(n')| <= w = 1.0 <= 1.5 <= c(n, n')
        - Therefore h(n) <= c(n, n') + h(n') for all edges (n, n') ✓

    Args:
        current: (floor, row, col) of the current node.
        goal: (floor, row, col) of the goal node.
        floor_weight: Weight multiplier for floor differences. Must be <= 1.5
                      to maintain admissibility. Default is 1.0.

    Returns:
        The heuristic estimate of the cost from current to goal.
    """
    f1, r1, c1 = current
    f2, r2, c2 = goal
    return abs(r1 - r2) + abs(c1 - c2) + floor_weight * abs(f1 - f2)


def euclidean_3d(current: Tuple[int, int, int], goal: Tuple[int, int, int],
                 floor_weight: float = 1.0) -> float:
    """
    3D Euclidean Distance heuristic for multi-floor navigation.

    h(n) = sqrt((x_n - x_g)^2 + (y_n - y_g)^2 + (w * (z_n - z_g))^2)

    Admissibility: Euclidean distance <= Manhattan distance for same weights,
    and Manhattan distance is already admissible, so Euclidean is also admissible.

    Args:
        current: (floor, row, col) of the current node.
        goal: (floor, row, col) of the goal node.
        floor_weight: Weight multiplier for floor differences. Default is 1.0.

    Returns:
        The heuristic estimate of the cost from current to goal.
    """
    f1, r1, c1 = current
    f2, r2, c2 = goal
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2 + (floor_weight * (f1 - f2)) ** 2)


def zero_heuristic(current: Tuple[int, int, int], goal: Tuple[int, int, int],
                   floor_weight: float = 1.0) -> float:
    """
    Zero heuristic — reduces A* to Dijkstra's algorithm.

    Always returns 0. Trivially admissible and consistent.
    Useful as a baseline comparison.

    Args:
        current: (floor, row, col) of the current node.
        goal: (floor, row, col) of the goal node.
        floor_weight: Ignored, kept for interface compatibility.

    Returns:
        0.0 always.
    """
    return 0.0
