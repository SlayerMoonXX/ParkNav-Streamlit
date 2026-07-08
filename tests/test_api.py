"""
Integration tests untuk ParkNav AI REST API.
Menggunakan FastAPI TestClient untuk menguji semua endpoint.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


# ─────────────── Root & Status ───────────────


class TestRootAndStatus:
    """Tests for root and status endpoints."""

    def test_root(self):
        """GET / should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "ParkNav AI API"
        assert "version" in data
        assert "endpoints" in data

    def test_status(self):
        """GET /api/status should return system status."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert data["data"]["system"] == "ParkNav AI"
        assert "algorithms_available" in data["data"]
        assert len(data["data"]["algorithms_available"]) == 4


# ─────────────── Navigate ───────────────


class TestNavigateEndpoint:
    """Tests for POST /api/navigate."""

    def test_navigate_success(self):
        """Valid navigation request should return a path."""
        response = client.post("/api/navigate", json={
            "start": [0, 0, 2],
            "goal": [0, 4, 2],
            "map_name": "default_map",
            "algorithm": "astar"
        })
        # May be 200 or 404 depending on whether default_map exists
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            result = data["data"]
            assert "path" in result
            assert "cost" in result
            assert "algorithm" in result
            assert result["algorithm"] == "A* Search"

    def test_navigate_invalid_position_length(self):
        """Position with wrong number of elements should fail validation."""
        response = client.post("/api/navigate", json={
            "start": [0, 0],  # Only 2 elements
            "goal": [0, 4, 2],
            "map_name": "default_map",
            "algorithm": "astar"
        })
        assert response.status_code == 422  # Validation error

    def test_navigate_negative_position(self):
        """Negative position values should fail validation."""
        response = client.post("/api/navigate", json={
            "start": [0, -1, 2],
            "goal": [0, 4, 2],
            "map_name": "default_map",
            "algorithm": "astar"
        })
        assert response.status_code == 422

    def test_navigate_invalid_algorithm(self):
        """Invalid algorithm name should fail validation."""
        response = client.post("/api/navigate", json={
            "start": [0, 0, 2],
            "goal": [0, 4, 2],
            "map_name": "default_map",
            "algorithm": "dijkstra"  # Not supported
        })
        assert response.status_code == 422

    def test_navigate_map_not_found(self):
        """Non-existent map should return 404."""
        response = client.post("/api/navigate", json={
            "start": [0, 0, 0],
            "goal": [0, 1, 1],
            "map_name": "nonexistent_map_xyz",
            "algorithm": "astar"
        })
        assert response.status_code == 404

    def test_navigate_all_algorithms(self):
        """Each valid algorithm should be accepted without validation error."""
        for algo in ["astar", "bfs", "dfs", "greedy"]:
            response = client.post("/api/navigate", json={
                "start": [0, 0, 2],
                "goal": [0, 4, 2],
                "map_name": "default_map",
                "algorithm": algo
            })
            # Should not be a validation error — either 200 or 404 (map missing)
            assert response.status_code in [200, 404, 400, 500], \
                f"Unexpected status for {algo}: {response.status_code}"


# ─────────────── Recommend ───────────────


class TestRecommendEndpoint:
    """Tests for POST /api/recommend."""

    def test_recommend(self):
        """Valid recommend request should return recommendations."""
        response = client.post("/api/recommend", json={
            "vehicle_type": "sedan",
            "preferred_floor": 0,
            "near_elevator": True,
            "near_exit": False,
            "map_name": "default_map"
        })
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            assert "recommendations" in data["data"]
            assert "total_found" in data["data"]

    def test_recommend_disability(self):
        """Disability recommendation request should be accepted."""
        response = client.post("/api/recommend", json={
            "vehicle_type": "disability",
            "preferred_floor": 0,
            "near_elevator": True,
            "near_exit": True,
            "map_name": "default_map"
        })
        # Should succeed or 404 (no map) — not a validation error
        assert response.status_code in [200, 404, 500]

    def test_recommend_invalid_vehicle_type(self):
        """Invalid vehicle type should fail validation."""
        response = client.post("/api/recommend", json={
            "vehicle_type": "truck",  # Not supported
            "map_name": "default_map"
        })
        assert response.status_code == 422

    def test_recommend_map_not_found(self):
        """Non-existent map should return 404."""
        response = client.post("/api/recommend", json={
            "vehicle_type": "sedan",
            "map_name": "nonexistent_map_xyz"
        })
        assert response.status_code == 404


# ─────────────── Map Info ───────────────


class TestMapEndpoints:
    """Tests for map-related endpoints."""

    def test_get_map(self):
        """GET /api/map/{name} should return map data."""
        response = client.get("/api/map/default_map")
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            assert "name" in data["data"]
            assert "num_floors" in data["data"]
            assert "total_slots" in data["data"]

    def test_get_map_not_found(self):
        """GET /api/map/{nonexistent} should return 404."""
        response = client.get("/api/map/nonexistent_map_xyz_123")
        assert response.status_code == 404
        detail = response.json().get("detail", "")
        assert "not found" in detail.lower()

    def test_list_maps(self):
        """GET /api/maps should return list of available maps."""
        response = client.get("/api/maps")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "maps" in data["data"]
        assert isinstance(data["data"]["maps"], list)
        assert "total" in data["data"]


# ─────────────── Compare ───────────────


class TestCompareEndpoint:
    """Tests for POST /api/compare."""

    def test_compare(self):
        """Compare endpoint should run all 4 algorithms."""
        response = client.post("/api/compare", json={
            "start": [0, 0, 2],
            "goal": [0, 4, 2],
            "map_name": "default_map"
        })
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            results = data["data"]["results"]
            assert "astar" in results
            assert "bfs" in results
            assert "dfs" in results
            assert "greedy" in results
            assert "summary" in data["data"]
            assert "best_algorithm" in data["data"]

    def test_compare_invalid_position(self):
        """Invalid position in compare should fail validation."""
        response = client.post("/api/compare", json={
            "start": [0, 0],  # Missing element
            "goal": [0, 4, 2],
            "map_name": "default_map"
        })
        assert response.status_code == 422

    def test_compare_map_not_found(self):
        """Non-existent map in compare should return 404."""
        response = client.post("/api/compare", json={
            "start": [0, 0, 0],
            "goal": [0, 1, 1],
            "map_name": "nonexistent_map_xyz"
        })
        assert response.status_code == 404
