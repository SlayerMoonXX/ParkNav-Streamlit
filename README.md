# 🅿️ ParkNav AI — Smart Parking Navigation System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Sistem Navigasi Parkir Bertingkat Otomatis** menggunakan **A\* Search Algorithm** dan **Rule-Based Expert System**  
> Tugas Akhir Mata Kuliah Kecerdasan Buatan (UAS AI)

---

## 📋 Deskripsi

**ParkNav AI** adalah sistem navigasi cerdas untuk gedung parkir bertingkat yang mampu:

1. **Menemukan rute optimal** dari titik masuk ke slot parkir menggunakan algoritma A\* Search.
2. **Merekomendasikan slot parkir terbaik** berdasarkan preferensi pengguna melalui Rule-Based Expert System.
3. **Membandingkan performa** empat algoritma pencarian: A\*, BFS, DFS, dan Greedy Best-First Search.

Sistem ini memodelkan gedung parkir sebagai graf 3D (lantai × baris × kolom) di mana setiap sel memiliki tipe tertentu (jalan, dinding, slot, ramp, elevator, dsb.) dan menggunakan teknik AI untuk navigasi yang efisien.

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    🖥️ PRESENTATION LAYER                │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │   Streamlit Web App  │  │   FastAPI REST Server    │  │
│  │   (app.py)           │  │   (src/api.py)           │  │
│  └──────────┬──────────┘  └─────────────┬────────────┘  │
├─────────────┼───────────────────────────┼───────────────┤
│             │     🧠 AI / LOGIC LAYER   │               │
│  ┌──────────▼───────────────────────────▼────────────┐  │
│  │                                                    │  │
│  │  ┌──────────────┐  ┌────────────────────────────┐  │  │
│  │  │  A* Search   │  │  Rule-Based Expert System  │  │  │
│  │  │  BFS / DFS   │  │  (Slot Recommendation)     │  │  │
│  │  │  Greedy      │  │                            │  │  │
│  │  └──────┬───────┘  └────────────┬───────────────┘  │  │
│  │         │                       │                  │  │
│  │  ┌──────▼───────────────────────▼───────────────┐  │  │
│  │  │          Heuristic Functions                 │  │  │
│  │  │          (Manhattan 3D Distance)             │  │  │
│  │  └─────────────────────┬────────────────────────┘  │  │
│  └────────────────────────┼───────────────────────────┘  │
├───────────────────────────┼─────────────────────────────┤
│              📦 DATA LAYER│                              │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │  ParkingMap Model (.json)   │  Expert Rules (.json) │  │
│  │  maps/default_map.json      │  models/expert_rules  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 Teknik AI yang Digunakan

### 1. A\* Search Algorithm (Informed Search)

A\* adalah algoritma pencarian informed yang menggabungkan:

- **g(n)**: Biaya aktual dari start ke node `n`
- **h(n)**: Estimasi biaya dari node `n` ke goal (heuristik)
- **f(n) = g(n) + h(n)**: Total estimasi biaya rute melalui `n`

**Heuristik yang digunakan**: 3D Manhattan Distance

```
h(n) = |x₁ - x₂| + |y₁ - y₂| + |floor₁ - floor₂| × floor_weight
```

di mana `floor_weight = 1.0` (bobot perpindahan antar lantai).

#### Pembuktian Admissibility (h(n) ≤ h\*(n))

- Jarak Manhattan selalu ≤ jarak sebenarnya pada grid (karena tidak ada diagonal)
- `floor_weight = 1.0` < minimum cost perpindahan antar lantai via ramp (`1.5`)
- Oleh karena itu, `h(n)` **tidak pernah overestimate** → **Admissible** ✅

#### Pembuktian Consistency (Triangle Inequality)

Untuk setiap edge `(n, n')` dengan cost `c(n, n')`:

```
h(n) ≤ c(n, n') + h(n')
⟺ h(n) - h(n') ≤ c(n, n')
```

- Perpindahan horizontal/vertikal: `|h(n) - h(n')| ≤ 1 ≤ c(n, n') = 1.0` ✅
- Perpindahan antar lantai (ramp): `|h(n) - h(n')| ≤ 1.0 ≤ c(n, n') = 1.5` ✅

→ Heuristik **Consistent** ✅

#### Jaminan A\*

| Properti    | Status | Penjelasan                                        |
| ----------- | ------ | ------------------------------------------------- |
| **Optimal** | ✅     | Selalu menemukan rute dengan biaya minimum         |
| **Complete**| ✅     | Selalu menemukan solusi jika ada                   |
| **Efisien** | ✅     | Mengeksplorasi lebih sedikit node dibanding BFS    |

### 2. Rule-Based Expert System

Sistem pakar berbasis aturan untuk rekomendasi slot parkir berdasarkan preferensi pengguna.

#### Aturan (Rules):

| # | Aturan                | Deskripsi                                                           |
|---|----------------------|---------------------------------------------------------------------|
| 1 | Vehicle Size Match   | Cocokkan ukuran slot dengan jenis kendaraan (sedan, SUV, motor)     |
| 2 | Floor Preference     | Prioritaskan lantai bawah untuk akses cepat                         |
| 3 | Elevator Proximity   | Dekatkan slot ke elevator untuk kenyamanan                          |
| 4 | Exit Proximity       | Dekatkan slot ke pintu keluar untuk kemudahan keluar                |
| 5 | Accessibility        | Prioritaskan slot aksesibel untuk pengguna disabilitas              |

Setiap aturan memberikan **skor** pada setiap slot yang tersedia. Slot dengan skor tertinggi direkomendasikan.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9** or higher
- **pip** (Python package manager)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/parking-nav-ai.git
cd parking-nav-ai

# (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### Run API Server

```bash
uvicorn src.api:app --reload --port 8000
```

Dokumentasi API otomatis tersedia di `http://localhost:8000/docs` (Swagger UI).

### Run Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
parking-nav-ai/
├── app.py                     # 🖥️ Streamlit web application
├── requirements.txt           # 📦 Python dependencies
├── README.md                  # 📖 Project documentation
├── .gitignore                 # 🚫 Git ignore rules
│
├── src/                       # 🧠 Core source code
│   ├── __init__.py
│   ├── parking_map.py         # ParkingMap model & CellType enum
│   ├── astar.py               # A*, BFS, DFS, Greedy search implementations
│   ├── heuristic.py           # Heuristic functions (Manhattan 3D)
│   ├── expert_system.py       # Rule-Based Expert System
│   ├── utils.py               # Utility functions (load_map, etc.)
│   └── api.py                 # FastAPI REST API server
│
├── maps/                      # 🗺️ Parking map definitions (JSON)
│   └── default_map.json
│
├── models/                    # 🤖 AI model configurations
│   └── expert_rules.json      # Expert system rules
│
└── tests/                     # 🧪 Test suite
    ├── __init__.py
    ├── test_astar.py
    ├── test_parking_map.py
    ├── test_expert_system.py
    └── test_api.py
```

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### `GET /api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

#### `POST /api/navigate`

Find the optimal path between two points in the parking structure.

**Request Body:**
```json
{
  "start": [0, 0, 2],
  "goal": [1, 2, 9],
  "map_name": "default_map",
  "algorithm": "astar"
}
```

| Field       | Type     | Required | Description                                      |
| ----------- | -------- | -------- | ------------------------------------------------ |
| `start`     | `[int, int, int]` | ✅ | Start position `[floor, row, col]`        |
| `goal`      | `[int, int, int]` | ✅ | Goal position `[floor, row, col]`         |
| `map_name`  | `string` | ❌       | Map name (default: `"default_map"`)              |
| `algorithm` | `string` | ❌       | `"astar"`, `"bfs"`, `"dfs"`, `"greedy"` (default: `"astar"`) |

**Response (Success):**
```json
{
  "success": true,
  "path": [[0, 0, 2], [0, 1, 2], [0, 2, 2], [0, 2, 3]],
  "cost": 3.0,
  "nodes_explored": 45,
  "nodes_generated": 120,
  "execution_time": 2.3,
  "algorithm": "astar",
  "message": "Path found successfully"
}
```

**Response (Failure):**
```json
{
  "success": false,
  "path": [],
  "cost": 0,
  "nodes_explored": 200,
  "nodes_generated": 500,
  "execution_time": 15.1,
  "algorithm": "astar",
  "message": "No path found between start and goal"
}
```

---

#### `POST /api/recommend`

Get slot recommendations from the expert system.

**Request Body:**
```json
{
  "map_name": "default_map",
  "preferences": {
    "vehicle_type": "sedan",
    "prefer_lower_floor": true,
    "near_elevator": false,
    "near_exit": true,
    "accessibility": false
  }
}
```

| Field          | Type     | Required | Description                            |
| -------------- | -------- | -------- | -------------------------------------- |
| `map_name`     | `string` | ❌       | Map name (default: `"default_map"`)    |
| `preferences`  | `object` | ✅       | User preference dictionary             |

**Response:**
```json
{
  "recommendations": [
    {
      "slot_id": "A1-03",
      "floor": 0,
      "row": 2,
      "col": 9,
      "score": 92.5,
      "reasons": [
        "Cocok untuk sedan",
        "Lantai bawah — akses cepat",
        "Dekat pintu keluar"
      ],
      "slot_type": "standard"
    }
  ]
}
```

---

#### `GET /api/maps`

List all available parking maps.

**Response:**
```json
{
  "maps": ["default_map", "large_mall", "hospital"]
}
```

---

#### `GET /api/maps/{map_name}`

Get details of a specific parking map.

**Response:**
```json
{
  "name": "default_map",
  "num_floors": 3,
  "num_rows": 8,
  "num_cols": 12,
  "floor_names": ["Lantai 1", "Lantai 2", "Lantai 3"],
  "total_slots": 48,
  "available_slots": 32,
  "occupied_slots": 16
}
```

---

#### `POST /api/compare`

Compare multiple search algorithms on the same start/goal.

**Request Body:**
```json
{
  "start": [0, 0, 2],
  "goal": [1, 2, 9],
  "map_name": "default_map"
}
```

**Response:**
```json
{
  "results": {
    "astar": {
      "success": true,
      "cost": 12.5,
      "nodes_explored": 45,
      "execution_time": 2.3
    },
    "bfs": {
      "success": true,
      "cost": 12.5,
      "nodes_explored": 120,
      "execution_time": 5.1
    },
    "dfs": {
      "success": true,
      "cost": 18.0,
      "nodes_explored": 35,
      "execution_time": 1.8
    },
    "greedy": {
      "success": true,
      "cost": 14.0,
      "nodes_explored": 28,
      "execution_time": 1.2
    }
  }
}
```

---

## 📊 Evaluasi

### Perbandingan Algoritma

| Metric          | A\*   | BFS   | DFS   | Greedy |
| --------------- | ----- | ----- | ----- | ------ |
| **Optimal**     | ✅    | ✅\*  | ❌    | ❌     |
| **Complete**    | ✅    | ✅    | ❌    | ✅     |
| **Efisien**     | ✅    | ❌    | ❌    | ✅     |
| **Informed**    | ✅    | ❌    | ❌    | ✅     |
| **Time Complexity**  | O(b^d) | O(b^d) | O(b^m) | O(b^d) |
| **Space Complexity** | O(b^d) | O(b^d) | O(bm) | O(b^d) |

> \*BFS optimal hanya untuk graf dengan bobot seragam (unweighted).

### Analisis Performa

- **A\* Search**: Performa terbaik secara keseluruhan — optimal dan efisien berkat heuristik yang admissible dan consistent.
- **BFS**: Menjamin solusi optimal pada graf unweighted, namun mengeksplorasi terlalu banyak node.
- **DFS**: Cepat dalam menemukan solusi tetapi **tidak menjamin** solusi optimal. Dapat terjebak di cabang yang dalam.
- **Greedy Best-First**: Cepat tetapi hanya mempertimbangkan heuristik tanpa biaya aktual, sehingga tidak optimal.

---

## 🛠️ Teknologi

| Komponen         | Teknologi                            |
| ---------------- | ------------------------------------ |
| **Language**     | Python 3.9+                          |
| **Web UI**       | Streamlit 1.28+                      |
| **REST API**     | FastAPI 0.104+ + Uvicorn             |
| **Visualization**| Matplotlib + Plotly                   |
| **Testing**      | Pytest                               |
| **Data Format**  | JSON                                 |
| **Validation**   | Pydantic 2.0+                        |

---

## 👥 Tim Pengembang

| Nama              | NIM         | Peran                        |
| ----------------- | ----------- | ---------------------------- |
| Anggota 1         | XXXXXXXXXX  | Lead Developer & AI Engineer |
| Anggota 2         | XXXXXXXXXX  | Backend & API Developer      |
| Anggota 3         | XXXXXXXXXX  | Frontend & Testing           |

---

## 📚 Referensi

1. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107.
3. Streamlit Documentation. https://docs.streamlit.io/
4. FastAPI Documentation. https://fastapi.tiangolo.com/

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ParkNav AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Made with ❤️ for UAS AI
</p>
