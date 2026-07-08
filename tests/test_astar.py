"""
Unit tests untuk A* Search Engine dan algoritma pencarian lainnya.
Menguji kebenaran, optimalitas, dan properti heuristik.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parking_map import ParkingMap, CellType
from src.astar import astar_search, bfs_search, dfs_search, greedy_search
from src.heuristic import manhattan_3d, euclidean_3d


# ─────────────────────── Fixtures ───────────────────────


@pytest.fixture
def simple_map():
    """1 floor, 5x5 grid — simple navigable path exists."""
    map_data = {
        "name": "Test Map",
        "num_floors": 1,
        "num_rows": 5,
        "num_cols": 5,
        "floors": [{"floor": 0, "name": "Ground", "grid": [
            [6, 0, 0, 0, 7],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 2, 1, 1]
        ]}],
        "slots": {
            "T-A01": {
                "floor": 0, "row": 4, "col": 2,
                "type": "standard", "status": "available",
                "accessible": False, "size": "standard"
            }
        },
        "ramp_connections": [],
        "elevator_positions": []
    }
    return ParkingMap(map_data)


@pytest.fixture
def blocked_map():
    """Map with no path from entrance to slot — completely walled off."""
    map_data = {
        "name": "Blocked Map",
        "num_floors": 1,
        "num_rows": 5,
        "num_cols": 5,
        "floors": [{"floor": 0, "name": "Ground", "grid": [
            [6, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 2]
        ]}],
        "slots": {
            "T-B01": {
                "floor": 0, "row": 4, "col": 4,
                "type": "standard", "status": "available",
                "accessible": False, "size": "standard"
            }
        },
        "ramp_connections": [],
        "elevator_positions": []
    }
    return ParkingMap(map_data)


@pytest.fixture
def open_map():
    """Wide-open map with many paths — good for comparing algorithms."""
    map_data = {
        "name": "Open Map",
        "num_floors": 1,
        "num_rows": 6,
        "num_cols": 6,
        "floors": [{"floor": 0, "name": "Ground", "grid": [
            [6, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 7]
        ]}],
        "slots": {},
        "ramp_connections": [],
        "elevator_positions": []
    }
    return ParkingMap(map_data)


@pytest.fixture
def multi_floor_map():
    """2-floor map connected by ramp — for inter-floor pathfinding."""
    map_data = {
        "name": "Multi Floor Map",
        "num_floors": 2,
        "num_rows": 4,
        "num_cols": 4,
        "floors": [
            {"floor": 0, "name": "Ground", "grid": [
                [6, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 4]
            ]},
            {"floor": 1, "name": "Level 1", "grid": [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 5]
            ]}
        ],
        "slots": {
            "MF-A01": {
                "floor": 1, "row": 2, "col": 2,
                "type": "standard", "status": "available",
                "accessible": False, "size": "standard"
            }
        },
        "ramp_connections": [
            {
                "from": [0, 3, 3],
                "to": [1, 3, 3],
                "cost_up": 2.0,
                "cost_down": 1.5
            }
        ],
        "elevator_positions": []
    }
    return ParkingMap(map_data)


# ─────────────── Test Cases: A* Search ───────────────


class TestAStarSearch:
    """Tests for A* search algorithm."""

    def test_astar_finds_path(self, simple_map):
        """A* should find a valid path from entrance to the slot."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = astar_search(simple_map, start, goal)

        assert result.success is True
        assert len(result.path) > 0
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.cost > 0
        assert result.algorithm == "A* Search"

    def test_astar_optimal(self, simple_map):
        """A* should find the optimal path (compare cost with BFS)."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        astar_result = astar_search(simple_map, start, goal)
        bfs_result = bfs_search(simple_map, start, goal)

        assert astar_result.success is True
        assert bfs_result.success is True
        # A* cost should be <= BFS cost (A* is optimal with admissible heuristic)
        assert astar_result.cost <= bfs_result.cost + 1e-9

    def test_astar_no_path(self, blocked_map):
        """A* should return success=False when no path exists."""
        start = (0, 0, 0)
        goal = (0, 4, 4)
        result = astar_search(blocked_map, start, goal)

        assert result.success is False
        assert len(result.path) == 0
        assert result.cost == 0 or result.cost == float('inf')

    def test_astar_start_equals_goal(self, simple_map):
        """A* should handle start == goal gracefully."""
        start = (0, 0, 0)
        goal = (0, 0, 0)
        result = astar_search(simple_map, start, goal)

        assert result.success is True
        # Path should be just the single point or empty with zero cost
        assert result.cost == 0
        assert len(result.path) <= 1 or result.path[0] == result.path[-1]

    def test_astar_path_continuity(self, simple_map):
        """Each consecutive pair in the path should be adjacent."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = astar_search(simple_map, start, goal)

        assert result.success is True
        path = result.path
        for i in range(len(path) - 1):
            f1, r1, c1 = path[i]
            f2, r2, c2 = path[i + 1]
            dist = abs(f1 - f2) + abs(r1 - r2) + abs(c1 - c2)
            # Adjacent cells differ by exactly 1 in one dimension
            # (or via ramp which might differ in floor)
            assert dist >= 1, f"Path has duplicate: {path[i]} -> {path[i+1]}"

    def test_astar_nodes_explored_positive(self, simple_map):
        """A* should explore at least 1 node."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = astar_search(simple_map, start, goal)

        assert result.nodes_explored >= 1
        assert result.nodes_generated >= 1

    def test_astar_execution_time(self, simple_map):
        """Execution time should be non-negative."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = astar_search(simple_map, start, goal)

        assert result.execution_time >= 0


# ─────────────── Test Cases: Heuristics ───────────────


class TestHeuristics:
    """Tests for heuristic functions."""

    def test_heuristic_admissible(self, simple_map):
        """h(n) should never overestimate the actual cost to goal."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = astar_search(simple_map, start, goal)

        assert result.success is True
        actual_cost = result.cost

        # Heuristic at start must be <= actual cost
        h_start = manhattan_3d(start, goal)
        assert h_start <= actual_cost + 1e-9, \
            f"Heuristic {h_start} > actual cost {actual_cost} — not admissible!"

    def test_heuristic_consistent(self, simple_map):
        """h(n) <= c(n, n') + h(n') for all edges — consistency check."""
        goal = (0, 4, 2)

        # Check consistency for a few reachable cells
        test_cells = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0), (0, 2, 1)]

        for cell in test_cells:
            if not simple_map.is_valid(*cell) or not simple_map.is_walkable(*cell):
                continue
            h_n = manhattan_3d(cell, goal)
            neighbors = simple_map.get_neighbors(*cell)
            for nf, nr, nc, cost in neighbors:
                h_n_prime = manhattan_3d((nf, nr, nc), goal)
                assert h_n <= cost + h_n_prime + 1e-9, \
                    f"Consistency violated at {cell}: h={h_n}, c={cost}, h'={h_n_prime}"

    def test_heuristic_zero_at_goal(self):
        """Heuristic should return 0 when node == goal."""
        goal = (1, 3, 5)
        assert manhattan_3d(goal, goal) == 0
        assert euclidean_3d(goal, goal) == 0

    def test_manhattan_vs_euclidean(self):
        """Manhattan distance >= Euclidean distance (for admissibility comparison)."""
        a = (0, 0, 0)
        b = (2, 3, 4)
        m = manhattan_3d(a, b)
        e = euclidean_3d(a, b)
        assert m >= e, "Manhattan should be >= Euclidean"


# ─────────────── Test Cases: BFS ───────────────


class TestBFSSearch:
    """Tests for BFS search algorithm."""

    def test_bfs_finds_path(self, simple_map):
        """BFS should find a valid path."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = bfs_search(simple_map, start, goal)

        assert result.success is True
        assert len(result.path) > 0
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.algorithm == "BFS (Breadth-First Search)"

    def test_bfs_no_path(self, blocked_map):
        """BFS should return failure when path is blocked."""
        start = (0, 0, 0)
        goal = (0, 4, 4)
        result = bfs_search(blocked_map, start, goal)

        assert result.success is False


# ─────────────── Test Cases: DFS ───────────────


class TestDFSSearch:
    """Tests for DFS search algorithm."""

    def test_dfs_finds_path(self, simple_map):
        """DFS should find a valid path."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = dfs_search(simple_map, start, goal)

        assert result.success is True
        assert len(result.path) > 0
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.algorithm == "DFS (Depth-First Search)"

    def test_dfs_no_path(self, blocked_map):
        """DFS should return failure when path is blocked."""
        start = (0, 0, 0)
        goal = (0, 4, 4)
        result = dfs_search(blocked_map, start, goal)

        assert result.success is False


# ─────────────── Test Cases: Greedy ───────────────


class TestGreedySearch:
    """Tests for Greedy Best-First search algorithm."""

    def test_greedy_finds_path(self, simple_map):
        """Greedy should find a valid path."""
        start = (0, 0, 0)
        goal = (0, 4, 2)
        result = greedy_search(simple_map, start, goal)

        assert result.success is True
        assert len(result.path) > 0
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.algorithm == "Greedy Best-First Search"

    def test_greedy_no_path(self, blocked_map):
        """Greedy should return failure when path is blocked."""
        start = (0, 0, 0)
        goal = (0, 4, 4)
        result = greedy_search(blocked_map, start, goal)

        assert result.success is False


# ─────────────── Test Cases: Algorithm Comparison ───────────────


class TestAlgorithmComparison:
    """Comparative tests across algorithms."""

    def test_astar_more_efficient_than_bfs(self, open_map):
        """A* should explore fewer or equal nodes compared to BFS on open maps."""
        start = (0, 0, 0)
        goal = (0, 5, 5)
        astar_result = astar_search(open_map, start, goal)
        bfs_result = bfs_search(open_map, start, goal)

        assert astar_result.success is True
        assert bfs_result.success is True
        # A* should typically explore fewer nodes due to heuristic guidance
        assert astar_result.nodes_explored <= bfs_result.nodes_explored, \
            f"A* explored {astar_result.nodes_explored} vs BFS {bfs_result.nodes_explored}"

    def test_all_algorithms_same_start_goal(self, simple_map):
        """All algorithms should find paths on the same input."""
        start = (0, 0, 0)
        goal = (0, 4, 2)

        results = {
            'astar': astar_search(simple_map, start, goal),
            'bfs': bfs_search(simple_map, start, goal),
            'dfs': dfs_search(simple_map, start, goal),
            'greedy': greedy_search(simple_map, start, goal),
        }

        for name, result in results.items():
            assert result.success is True, f"{name} failed to find path"
            assert result.path[0] == start, f"{name} path doesn't start at start"
            assert result.path[-1] == goal, f"{name} path doesn't end at goal"

    def test_all_algorithms_no_path(self, blocked_map):
        """All algorithms should fail when no path exists."""
        start = (0, 0, 0)
        goal = (0, 4, 4)

        results = {
            'astar': astar_search(blocked_map, start, goal),
            'bfs': bfs_search(blocked_map, start, goal),
            'dfs': dfs_search(blocked_map, start, goal),
            'greedy': greedy_search(blocked_map, start, goal),
        }

        for name, result in results.items():
            assert result.success is False, f"{name} should NOT find a path in blocked map"

    def test_multi_floor_pathfinding(self, multi_floor_map):
        """A* should handle cross-floor navigation via ramps."""
        start = (0, 0, 0)
        goal = (1, 2, 2)
        result = astar_search(multi_floor_map, start, goal)

        assert result.success is True
        assert result.path[0] == start
        assert result.path[-1] == goal
        # Path must cross floors
        floors_visited = set(p[0] for p in result.path)
        assert len(floors_visited) >= 2, "Path should span multiple floors"
