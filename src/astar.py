"""Search algorithms for multi-story parking navigation.

Implements A*, BFS, DFS, and Greedy Best-First Search for pathfinding
in multi-floor parking structures.
"""

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

from .parking_map import ParkingMap
from .heuristic import manhattan_3d


@dataclass
class SearchResult:
    """Container for search algorithm results.

    Attributes:
        success: Whether a path was found.
        path: Ordered list of (floor, row, col) positions from start to goal.
        cost: Total path cost (sum of edge weights).
        nodes_explored: Number of nodes expanded (removed from frontier).
        nodes_generated: Number of nodes generated (added to frontier).
        execution_time: Search duration in milliseconds.
        algorithm: Name of the search algorithm used.
        message: Human-readable result message.
        exploration_order: Order in which nodes were explored, for visualization.
    """
    success: bool
    path: List[Tuple[int, int, int]]
    cost: float
    nodes_explored: int
    nodes_generated: int
    execution_time: float
    algorithm: str
    message: str
    exploration_order: List[Tuple[int, int, int]] = field(default_factory=list)


class Node:
    """Search tree node for pathfinding algorithms."""

    __slots__ = ('floor', 'row', 'col', 'g', 'h', 'f', 'parent')

    def __init__(self, floor: int, row: int, col: int,
                 g: float = 0.0, h: float = 0.0, parent: Optional['Node'] = None):
        self.floor = floor
        self.row = row
        self.col = col
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    @property
    def position(self) -> Tuple[int, int, int]:
        return (self.floor, self.row, self.col)

    def __lt__(self, other: 'Node') -> bool:
        if self.f == other.f:
            return self.h < other.h  # tie-break: prefer lower h
        return self.f < other.f

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.position == other.position

    def __hash__(self) -> int:
        return hash(self.position)


def _reconstruct_path(node: Node) -> List[Tuple[int, int, int]]:
    """Trace back from goal node to start to reconstruct the path."""
    path = []
    current = node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]


def _validate_inputs(parking_map: ParkingMap, start: tuple, goal: tuple) -> Optional[str]:
    """Validate search inputs and return error message if invalid, else None."""
    if len(start) != 3 or len(goal) != 3:
        return "Start and goal must be (floor, row, col) tuples of length 3"
    if not parking_map.is_valid(*start):
        return f"Start position {start} is out of bounds"
    if not parking_map.is_valid(*goal):
        return f"Goal position {goal} is out of bounds"
    if not parking_map.is_walkable(*start):
        return f"Start position {start} is not walkable"
    if not parking_map.is_walkable(*goal):
        return f"Goal position {goal} is not walkable"
    return None


def _make_failure_result(algorithm: str, message: str, start_time: float,
                         nodes_explored: int = 0, nodes_generated: int = 0,
                         exploration_order: Optional[List] = None) -> SearchResult:
    """Create a failure SearchResult."""
    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        success=False, path=[], cost=0.0,
        nodes_explored=nodes_explored, nodes_generated=nodes_generated,
        execution_time=elapsed, algorithm=algorithm, message=message,
        exploration_order=exploration_order or []
    )


def _make_trivial_result(algorithm: str, start: tuple, start_time: float) -> SearchResult:
    """Create a trivial SearchResult when start == goal."""
    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        success=True, path=[start], cost=0.0,
        nodes_explored=0, nodes_generated=1,
        execution_time=elapsed, algorithm=algorithm,
        message="Start is the goal"
    )


def astar_search(
    parking_map: ParkingMap,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    heuristic: Callable = manhattan_3d
) -> SearchResult:
    """A* Search — optimal pathfinding using f(n) = g(n) + h(n).

    Uses a priority queue ordered by f-value (total estimated cost).
    Guaranteed to find the optimal path when the heuristic is admissible
    and consistent.

    Args:
        parking_map: The parking structure to search.
        start: (floor, row, col) starting position.
        goal: (floor, row, col) goal position.
        heuristic: Heuristic function h(current, goal) -> float.

    Returns:
        SearchResult with the optimal path and search statistics.
    """
    start_time = time.perf_counter()
    algorithm = "A* Search"

    # Validate inputs
    error = _validate_inputs(parking_map, start, goal)
    if error:
        return _make_failure_result(algorithm, error, start_time)

    # Handle trivial case
    if start == goal:
        return _make_trivial_result(algorithm, start, start_time)

    start_node = Node(start[0], start[1], start[2], g=0.0, h=heuristic(start, goal))

    # Open list: priority queue of (f, counter, node) for stable tie-breaking
    counter = 0
    open_list = [(start_node.f, counter, start_node)]
    # Best g-value found for each position
    g_scores = {start: 0.0}
    closed_set = set()

    nodes_explored = 0
    nodes_generated = 1
    exploration_order = []

    while open_list:
        _, _, current = heapq.heappop(open_list)
        pos = current.position

        # Skip if already explored with a better cost
        if pos in closed_set:
            continue

        closed_set.add(pos)
        nodes_explored += 1
        exploration_order.append(pos)

        # Goal check
        if pos == goal:
            elapsed = (time.perf_counter() - start_time) * 1000
            path = _reconstruct_path(current)
            return SearchResult(
                success=True, path=path, cost=current.g,
                nodes_explored=nodes_explored, nodes_generated=nodes_generated,
                execution_time=elapsed, algorithm=algorithm,
                message=f"Path found with cost {current.g:.2f}",
                exploration_order=exploration_order
            )

        # Expand neighbors
        for n_floor, n_row, n_col, move_cost in parking_map.get_neighbors(*pos):
            neighbor_pos = (n_floor, n_row, n_col)
            if neighbor_pos in closed_set:
                continue

            tentative_g = current.g + move_cost

            # Only process if this is a better path
            if tentative_g < g_scores.get(neighbor_pos, float('inf')):
                g_scores[neighbor_pos] = tentative_g
                h = heuristic(neighbor_pos, goal)
                neighbor_node = Node(
                    n_floor, n_row, n_col,
                    g=tentative_g, h=h, parent=current
                )
                counter += 1
                heapq.heappush(open_list, (neighbor_node.f, counter, neighbor_node))
                nodes_generated += 1

    # No path found
    return _make_failure_result(
        algorithm, "No path found from start to goal", start_time,
        nodes_explored, nodes_generated, exploration_order
    )


def bfs_search(
    parking_map: ParkingMap,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int]
) -> SearchResult:
    """Breadth-First Search — finds the path with fewest steps.

    Uses a FIFO queue. Finds the shortest path by number of moves
    (unweighted), but does NOT consider edge weights, so it may not
    find the lowest-cost path in a weighted graph.

    Args:
        parking_map: The parking structure to search.
        start: (floor, row, col) starting position.
        goal: (floor, row, col) goal position.

    Returns:
        SearchResult with the shortest (by steps) path.
    """
    start_time = time.perf_counter()
    algorithm = "BFS (Breadth-First Search)"

    error = _validate_inputs(parking_map, start, goal)
    if error:
        return _make_failure_result(algorithm, error, start_time)

    if start == goal:
        return _make_trivial_result(algorithm, start, start_time)

    start_node = Node(start[0], start[1], start[2], g=0.0)
    queue = deque([start_node])
    visited = {start}

    nodes_explored = 0
    nodes_generated = 1
    exploration_order = []

    while queue:
        current = queue.popleft()
        pos = current.position
        nodes_explored += 1
        exploration_order.append(pos)

        # Goal check
        if pos == goal:
            elapsed = (time.perf_counter() - start_time) * 1000
            path = _reconstruct_path(current)
            return SearchResult(
                success=True, path=path, cost=current.g,
                nodes_explored=nodes_explored, nodes_generated=nodes_generated,
                execution_time=elapsed, algorithm=algorithm,
                message=f"Path found with cost {current.g:.2f}",
                exploration_order=exploration_order
            )

        for n_floor, n_row, n_col, move_cost in parking_map.get_neighbors(*pos):
            neighbor_pos = (n_floor, n_row, n_col)
            if neighbor_pos not in visited:
                visited.add(neighbor_pos)
                neighbor_node = Node(
                    n_floor, n_row, n_col,
                    g=current.g + move_cost,
                    parent=current
                )
                queue.append(neighbor_node)
                nodes_generated += 1

    return _make_failure_result(
        algorithm, "No path found from start to goal", start_time,
        nodes_explored, nodes_generated, exploration_order
    )


def dfs_search(
    parking_map: ParkingMap,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int]
) -> SearchResult:
    """Depth-First Search — explores as deeply as possible first.

    Uses a LIFO stack. May NOT find the optimal or shortest path.
    Memory-efficient but can get trapped in deep branches.
    Useful as a comparison baseline to demonstrate A*'s superiority.

    Args:
        parking_map: The parking structure to search.
        start: (floor, row, col) starting position.
        goal: (floor, row, col) goal position.

    Returns:
        SearchResult with a path (not necessarily optimal).
    """
    start_time = time.perf_counter()
    algorithm = "DFS (Depth-First Search)"

    error = _validate_inputs(parking_map, start, goal)
    if error:
        return _make_failure_result(algorithm, error, start_time)

    if start == goal:
        return _make_trivial_result(algorithm, start, start_time)

    start_node = Node(start[0], start[1], start[2], g=0.0)
    stack = [start_node]
    visited = set()

    nodes_explored = 0
    nodes_generated = 1
    exploration_order = []

    while stack:
        current = stack.pop()
        pos = current.position

        if pos in visited:
            continue

        visited.add(pos)
        nodes_explored += 1
        exploration_order.append(pos)

        # Goal check
        if pos == goal:
            elapsed = (time.perf_counter() - start_time) * 1000
            path = _reconstruct_path(current)
            return SearchResult(
                success=True, path=path, cost=current.g,
                nodes_explored=nodes_explored, nodes_generated=nodes_generated,
                execution_time=elapsed, algorithm=algorithm,
                message=f"Path found with cost {current.g:.2f}",
                exploration_order=exploration_order
            )

        for n_floor, n_row, n_col, move_cost in parking_map.get_neighbors(*pos):
            neighbor_pos = (n_floor, n_row, n_col)
            if neighbor_pos not in visited:
                neighbor_node = Node(
                    n_floor, n_row, n_col,
                    g=current.g + move_cost,
                    parent=current
                )
                stack.append(neighbor_node)
                nodes_generated += 1

    return _make_failure_result(
        algorithm, "No path found from start to goal", start_time,
        nodes_explored, nodes_generated, exploration_order
    )


def greedy_search(
    parking_map: ParkingMap,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    heuristic: Callable = manhattan_3d
) -> SearchResult:
    """Greedy Best-First Search — uses only h(n), ignores g(n).

    Prioritizes nodes that appear closest to the goal. Typically faster
    than A* but does NOT guarantee an optimal path.

    Args:
        parking_map: The parking structure to search.
        start: (floor, row, col) starting position.
        goal: (floor, row, col) goal position.
        heuristic: Heuristic function h(current, goal) -> float.

    Returns:
        SearchResult with a path (not necessarily optimal).
    """
    start_time = time.perf_counter()
    algorithm = "Greedy Best-First Search"

    error = _validate_inputs(parking_map, start, goal)
    if error:
        return _make_failure_result(algorithm, error, start_time)

    if start == goal:
        return _make_trivial_result(algorithm, start, start_time)

    h_start = heuristic(start, goal)
    start_node = Node(start[0], start[1], start[2], g=0.0, h=h_start)

    counter = 0
    # Priority queue ordered by h only (not f = g + h)
    open_list = [(h_start, counter, start_node)]
    visited = set()

    nodes_explored = 0
    nodes_generated = 1
    exploration_order = []

    while open_list:
        _, _, current = heapq.heappop(open_list)
        pos = current.position

        if pos in visited:
            continue

        visited.add(pos)
        nodes_explored += 1
        exploration_order.append(pos)

        # Goal check
        if pos == goal:
            elapsed = (time.perf_counter() - start_time) * 1000
            path = _reconstruct_path(current)
            return SearchResult(
                success=True, path=path, cost=current.g,
                nodes_explored=nodes_explored, nodes_generated=nodes_generated,
                execution_time=elapsed, algorithm=algorithm,
                message=f"Path found with cost {current.g:.2f}",
                exploration_order=exploration_order
            )

        for n_floor, n_row, n_col, move_cost in parking_map.get_neighbors(*pos):
            neighbor_pos = (n_floor, n_row, n_col)
            if neighbor_pos not in visited:
                h = heuristic(neighbor_pos, goal)
                neighbor_node = Node(
                    n_floor, n_row, n_col,
                    g=current.g + move_cost,
                    h=h,
                    parent=current
                )
                counter += 1
                heapq.heappush(open_list, (h, counter, neighbor_node))
                nodes_generated += 1

    return _make_failure_result(
        algorithm, "No path found from start to goal", start_time,
        nodes_explored, nodes_generated, exploration_order
    )
