"""
REST API untuk ParkNav AI
FastAPI-based REST API untuk navigasi parkir multi-lantai
menggunakan algoritma A* Search dan Sistem Pakar.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
import traceback

from src.utils import load_map, get_available_maps
from src.astar import astar_search, bfs_search, dfs_search, greedy_search
from src.expert_system import ParkingExpertSystem, SlotRecommendation

app = FastAPI(
    title="ParkNav AI API",
    description="REST API for Multi-Story Parking Navigation using A* Search & Expert System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------- Pydantic Models ---------------

class NavigateRequest(BaseModel):
    start: List[int]  # [floor, row, col]
    goal: List[int]   # [floor, row, col]
    map_name: str = "default_map"
    algorithm: str = "astar"  # astar, bfs, dfs, greedy

    @validator('start', 'goal')
    def validate_position(cls, v):
        if len(v) != 3:
            raise ValueError('Position must have 3 elements: [floor, row, col]')
        if any(x < 0 for x in v):
            raise ValueError('Position values must be non-negative')
        return v

    @validator('algorithm')
    def validate_algorithm(cls, v):
        valid = ['astar', 'bfs', 'dfs', 'greedy']
        if v not in valid:
            raise ValueError(f'Algorithm must be one of {valid}')
        return v


class RecommendRequest(BaseModel):
    vehicle_type: str = "sedan"
    preferred_floor: Optional[int] = None
    near_elevator: bool = False
    near_exit: bool = False
    map_name: str = "default_map"

    @validator('vehicle_type')
    def validate_vehicle_type(cls, v):
        valid = ['sedan', 'suv', 'motorcycle', 'disability']
        if v not in valid:
            raise ValueError(f'Vehicle type must be one of {valid}')
        return v


class CompareRequest(BaseModel):
    start: List[int]
    goal: List[int]
    map_name: str = "default_map"

    @validator('start', 'goal')
    def validate_position(cls, v):
        if len(v) != 3:
            raise ValueError('Position must have 3 elements: [floor, row, col]')
        if any(x < 0 for x in v):
            raise ValueError('Position values must be non-negative')
        return v


# --------------- Algorithm Dispatcher ---------------

ALGORITHM_MAP = {
    'astar': astar_search,
    'bfs': bfs_search,
    'dfs': dfs_search,
    'greedy': greedy_search,
}


def _search_result_to_dict(result) -> dict:
    """Convert SearchResult dataclass to a JSON-serializable dict."""
    return {
        "success": result.success,
        "path": [list(p) for p in result.path],
        "cost": result.cost,
        "nodes_explored": result.nodes_explored,
        "nodes_generated": result.nodes_generated,
        "execution_time": result.execution_time,
        "algorithm": result.algorithm,
        "message": result.message,
        "path_length": len(result.path),
    }


def _recommendation_to_dict(rec: SlotRecommendation) -> dict:
    """Convert SlotRecommendation dataclass to a JSON-serializable dict."""
    return {
        "slot_id": rec.slot_id,
        "floor": rec.floor,
        "row": rec.row,
        "col": rec.col,
        "score": rec.score,
        "reasons": rec.reasons,
        "slot_type": rec.slot_type,
        "distance_to_entrance": rec.distance_to_entrance,
    }


# --------------- Endpoints ---------------

@app.get("/")
def root():
    """Root endpoint — API information."""
    return {
        "message": "ParkNav AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "navigate": "POST /api/navigate",
            "recommend": "POST /api/recommend",
            "compare": "POST /api/compare",
            "map": "GET /api/map/{map_name}",
            "maps": "GET /api/maps",
            "status": "GET /api/status",
        }
    }


@app.post("/api/navigate")
def navigate(request: NavigateRequest):
    """
    Navigate from start to goal using the selected algorithm.
    Returns the computed path, cost, and search statistics.
    """
    try:
        parking_map = load_map(request.map_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Map '{request.map_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map: {str(e)}")

    try:
        start = tuple(request.start)
        goal = tuple(request.goal)

        # Validate positions are within map bounds
        if not parking_map.is_valid(*start):
            raise HTTPException(
                status_code=400,
                detail=f"Start position {list(start)} is out of map bounds"
            )
        if not parking_map.is_valid(*goal):
            raise HTTPException(
                status_code=400,
                detail=f"Goal position {list(goal)} is out of map bounds"
            )

        search_fn = ALGORITHM_MAP[request.algorithm]
        result = search_fn(parking_map, start, goal)

        return {
            "status": "success",
            "data": _search_result_to_dict(result),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Navigation error: {str(e)}\n{traceback.format_exc()}"
        )


@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    """
    Get parking slot recommendations from the Expert System.
    """
    try:
        parking_map = load_map(request.map_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Map '{request.map_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map: {str(e)}")

    try:
        expert = ParkingExpertSystem()
        preferences = {
            'vehicle_type': request.vehicle_type,
            'preferred_floor': request.preferred_floor,
            'near_elevator': request.near_elevator,
            'near_exit': request.near_exit,
        }

        recommendations = expert.recommend(parking_map, preferences)

        return {
            "status": "success",
            "data": {
                "recommendations": [_recommendation_to_dict(r) for r in recommendations],
                "total_found": len(recommendations),
                "preferences": preferences,
                "rules_applied": expert.get_rules_description(),
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation error: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/map/{map_name}")
def get_map(map_name: str):
    """
    Get full map data for a specific map.
    """
    try:
        parking_map = load_map(map_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map: {str(e)}")

    try:
        available_slots = parking_map.get_available_slots()
        entrance = parking_map.get_entrance()
        exit_pos = parking_map.get_exit()

        return {
            "status": "success",
            "data": {
                "name": parking_map.name,
                "num_floors": parking_map.num_floors,
                "num_rows": parking_map.num_rows,
                "num_cols": parking_map.num_cols,
                "total_slots": len(parking_map.slots),
                "available_slots": len(available_slots),
                "entrance": list(entrance) if entrance else None,
                "exit": list(exit_pos) if exit_pos else None,
                "floors": parking_map.floors,
                "slots": parking_map.slots,
                "ramp_connections": parking_map.ramp_connections,
                "elevator_positions": parking_map.elevator_positions,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving map data: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/maps")
def list_maps():
    """
    List all available maps.
    """
    try:
        maps = get_available_maps()
        return {
            "status": "success",
            "data": {
                "maps": maps,
                "total": len(maps),
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing maps: {str(e)}\n{traceback.format_exc()}"
        )


@app.post("/api/compare")
def compare_algorithms(request: CompareRequest):
    """
    Run all 4 algorithms on the same start/goal and return comparison.
    """
    try:
        parking_map = load_map(request.map_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Map '{request.map_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map: {str(e)}")

    try:
        start = tuple(request.start)
        goal = tuple(request.goal)

        if not parking_map.is_valid(*start):
            raise HTTPException(
                status_code=400,
                detail=f"Start position {list(start)} is out of map bounds"
            )
        if not parking_map.is_valid(*goal):
            raise HTTPException(
                status_code=400,
                detail=f"Goal position {list(goal)} is out of map bounds"
            )

        results = {}
        for algo_name, algo_fn in ALGORITHM_MAP.items():
            try:
                result = algo_fn(parking_map, start, goal)
                results[algo_name] = _search_result_to_dict(result)
            except Exception as e:
                results[algo_name] = {
                    "success": False,
                    "error": str(e),
                    "algorithm": algo_name,
                }

        # Build summary comparison
        summary = []
        for algo_name, res in results.items():
            if res.get("success"):
                summary.append({
                    "algorithm": algo_name,
                    "cost": res["cost"],
                    "path_length": res.get("path_length", 0),
                    "nodes_explored": res["nodes_explored"],
                    "execution_time": res["execution_time"],
                })

        # Sort summary by cost then by nodes explored
        summary.sort(key=lambda x: (x["cost"], x["nodes_explored"]))

        return {
            "status": "success",
            "data": {
                "start": list(start),
                "goal": list(goal),
                "results": results,
                "summary": summary,
                "best_algorithm": summary[0]["algorithm"] if summary else None,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison error: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/status")
def status():
    """
    Return system status: available maps, total slots, etc.
    """
    try:
        available_maps = get_available_maps()

        maps_info = []
        total_slots = 0
        total_available = 0

        for map_name in available_maps:
            try:
                pm = load_map(map_name)
                slot_count = len(pm.slots)
                avail_count = len(pm.get_available_slots())
                total_slots += slot_count
                total_available += avail_count
                maps_info.append({
                    "name": map_name,
                    "floors": pm.num_floors,
                    "total_slots": slot_count,
                    "available_slots": avail_count,
                })
            except Exception:
                maps_info.append({
                    "name": map_name,
                    "error": "Could not load map",
                })

        return {
            "status": "success",
            "data": {
                "system": "ParkNav AI",
                "version": "1.0.0",
                "algorithms_available": list(ALGORITHM_MAP.keys()),
                "maps": maps_info,
                "total_maps": len(available_maps),
                "total_slots": total_slots,
                "total_available_slots": total_available,
                "expert_system": "active",
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Status error: {str(e)}\n{traceback.format_exc()}"
        )
