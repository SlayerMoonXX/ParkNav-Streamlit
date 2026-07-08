"""Multi-Story Parking Navigation AI - Core Engine.

This package provides A* search-based navigation for multi-story parking structures,
with support for multiple search algorithms and an expert system for slot recommendation.
"""

from .parking_map import ParkingMap, CellType
from .astar import astar_search, bfs_search, dfs_search, greedy_search, SearchResult
from .heuristic import manhattan_3d, euclidean_3d
from .expert_system import ParkingExpertSystem, SlotRecommendation
from .utils import load_map, get_available_maps

__version__ = '1.0.0'
__all__ = [
    'ParkingMap', 'CellType',
    'astar_search', 'bfs_search', 'dfs_search', 'greedy_search', 'SearchResult',
    'manhattan_3d', 'euclidean_3d',
    'ParkingExpertSystem',
    'load_map', 'get_available_maps',
]
