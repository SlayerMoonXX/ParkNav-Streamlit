"""Utility functions for loading maps and formatting output."""

import os
import json
from typing import List


def load_map(map_name: str) -> 'ParkingMap':
    """Load a parking map by name from the data/maps/ directory.

    Args:
        map_name: Name of the map (without .json extension).

    Returns:
        ParkingMap instance.

    Raises:
        FileNotFoundError: If the map file doesn't exist.
    """
    from .parking_map import ParkingMap
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, 'data', 'maps', f'{map_name}.json')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Map '{map_name}' not found at {filepath}")
    return ParkingMap.from_json(filepath)


def get_available_maps() -> List[str]:
    """List all available map names from the data/maps/ directory.

    Returns:
        Sorted list of map names (without .json extension).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(base_dir, 'data', 'maps')
    if not os.path.isdir(maps_dir):
        return []
    maps = [
        f.replace('.json', '')
        for f in os.listdir(maps_dir)
        if f.endswith('.json')
    ]
    return sorted(maps)


def format_path(path: list) -> str:
    """Format a navigation path for human-readable display.

    Args:
        path: List of (floor, row, col) tuples.

    Returns:
        Formatted multi-line string showing the path with floor transitions.
    """
    if not path:
        return "No path"

    lines = [f"Path ({len(path)} steps):"]
    current_floor = None

    for i, (floor, row, col) in enumerate(path):
        if floor != current_floor:
            current_floor = floor
            lines.append(f"  --- Floor {floor + 1} ---")
        if i == 0:
            prefix = "  START → "
        elif i == len(path) - 1:
            prefix = "  GOAL  → "
        else:
            prefix = "          "
        lines.append(f"{prefix}({row}, {col})")

    return "\n".join(lines)


def format_time(ms: float) -> str:
    """Format milliseconds to a human-readable time string.

    Args:
        ms: Time in milliseconds.

    Returns:
        Formatted string (e.g., '0.42 ms', '1.23 s', '2 min 5.00 s').
    """
    if ms < 1.0:
        return f"{ms * 1000:.0f} us"
    elif ms < 1000.0:
        return f"{ms:.2f} ms"
    elif ms < 60000.0:
        return f"{ms / 1000:.2f} s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes} min {seconds:.2f} s"


def format_search_result(result) -> str:
    """Format a SearchResult object for display.

    Args:
        result: A SearchResult instance.

    Returns:
        Multi-line formatted string with search statistics.
    """
    status = '[v] Path Found' if result.success else '[x] No Path Found'
    sep = '+' + '=' * 38 + '+'
    lines = [
        sep,
        f"|  {result.algorithm:^34s}  |",
        sep,
        f"|  Status:    {status:>24s}  |",
        f"|  Cost:      {result.cost:>24.2f}  |",
        f"|  Steps:     {len(result.path):>24d}  |",
        f"|  Explored:  {result.nodes_explored:>24d}  |",
        f"|  Generated: {result.nodes_generated:>24d}  |",
        f"|  Time:      {format_time(result.execution_time):>24s}  |",
        sep,
    ]
    return "\n".join(lines)
