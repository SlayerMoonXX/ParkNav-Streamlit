"""
🅿️ ParkNav AI — Smart Parking Navigation System
Main Streamlit Application

Sistem Navigasi Parkir Bertingkat Otomatis menggunakan
A* Search Algorithm dan Rule-Based Expert System.
"""

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import time

# --- Local imports ---
from src.parking_map import ParkingMap, CellType
from src.astar import astar_search, bfs_search, dfs_search, greedy_search, SearchResult
from src.heuristic import manhattan_3d
from src.expert_system import ParkingExpertSystem, SlotRecommendation
from src.utils import load_map, get_available_maps

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🅿️ ParkNav AI — Smart Parking Navigation",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global dark theme with gradient ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a35 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e1e3f, #2d2d5e);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(80,80,200,0.2);
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0 4px 0;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: #a0a0c0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.metric-card .metric-icon {
    font-size: 1.6rem;
    margin-bottom: 4px;
}

/* ── Success & error banners ── */
.success-banner {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    border-radius: 12px;
    padding: 18px 24px;
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 0 4px 20px rgba(0,176,155,0.3);
    margin-bottom: 16px;
}
.error-banner {
    background: linear-gradient(135deg, #e53e3e, #c53030);
    border-radius: 12px;
    padding: 18px 24px;
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 0 4px 20px rgba(229,62,62,0.3);
    margin-bottom: 16px;
}

/* ── Recommendation cards ── */
.rec-card {
    background: linear-gradient(135deg, #1a1a3e, #252560);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.rec-card:hover {
    transform: translateY(-2px);
}
.rec-card .rec-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 8px;
}
.rec-card .rec-score {
    font-size: 0.9rem;
    color: #94a3b8;
}
.rec-card .rec-reason {
    color: #cbd5e1;
    font-size: 0.88rem;
    padding-left: 12px;
    border-left: 3px solid #6366f1;
    margin: 6px 0;
}

/* ── Score bar ── */
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 10px;
    margin: 8px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    transition: width 0.6s ease;
}

/* ── Path step styling ── */
.path-step {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px 0;
    border-left: 3px solid #6366f1;
    color: #c4c4e0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.9rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.04);
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    color: #a0a0c0;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* ── Dividers ── */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #a78bfa, transparent);
    border: none;
    margin: 20px 0;
}

/* ── Section title ── */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}

/* ── Hide default Streamlit header ── */
header[data-testid="stHeader"] {
    background: transparent;
}

/* ── Button styling ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99,102,241,0.5) !important;
}

/* ── About section ── */
.about-block {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 16px;
    color: #c4c4e0;
    line-height: 1.7;
}
.about-block h3 {
    color: #a78bfa;
    margin-bottom: 12px;
}
.about-block code {
    background: rgba(99,102,241,0.15);
    padding: 2px 6px;
    border-radius: 4px;
    color: #c4b5fd;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_parking_floor(parking_map, floor, path=None, explored=None, start=None, goal=None):
    """
    Render a single floor of the parking map as a matplotlib figure.

    Args:
        parking_map: ParkingMap object
        floor: Floor index to render
        path: Optional list of (floor, row, col) tuples — the found path
        explored: Optional list of (floor, row, col) — explored nodes
        start: Optional (floor, row, col) — start position
        goal: Optional (floor, row, col) — goal position

    Returns:
        matplotlib Figure
    """
    rows = parking_map.num_rows
    cols = parking_map.num_cols

    fig_width = max(10, cols * 0.9)
    fig_height = max(6, rows * 0.9)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Dark background
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#12122a')

    # Draw each cell
    for r in range(rows):
        for c in range(cols):
            cell_type = parking_map.get_cell(floor, r, c)
            color = CellType.COLORS.get(cell_type, '#333333')

            rect = plt.Rectangle((c, rows - 1 - r), 1, 1,
                                 linewidth=0.5,
                                 edgecolor='#2a2a4a',
                                 facecolor=color,
                                 alpha=0.85)
            ax.add_patch(rect)

            # Show slot IDs on available/occupied slots
            if cell_type in (CellType.SLOT_AVAILABLE, CellType.SLOT_OCCUPIED):
                slot_label = None
                for sid, sinfo in parking_map.slots.items():
                    pos = parking_map.get_slot_position(sid)
                    if pos and pos[0] == floor and pos[1] == r and pos[2] == c:
                        slot_label = sid
                        break
                if slot_label:
                    ax.text(c + 0.5, rows - 1 - r + 0.5, slot_label,
                            ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            color='white' if cell_type == CellType.SLOT_AVAILABLE else '#ffcccc')

            # Show cell-type labels for special cells
            elif cell_type in (CellType.RAMP_UP, CellType.RAMP_DOWN, CellType.ENTRANCE,
                               CellType.EXIT, CellType.ELEVATOR):
                label = CellType.NAMES.get(cell_type, '')[0]  # first letter
                ax.text(c + 0.5, rows - 1 - r + 0.5, label,
                        ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')

    # Draw explored nodes overlay
    if explored:
        for (ef, er, ec) in explored:
            if ef == floor:
                rect = plt.Rectangle((ec, rows - 1 - er), 1, 1,
                                     facecolor='#667eea', alpha=0.18,
                                     linewidth=0)
                ax.add_patch(rect)

    # Draw path
    if path:
        floor_path = [(r, c) for (f, r, c) in path if f == floor]
        if floor_path:
            # Draw path cells
            for (pr, pc) in floor_path:
                rect = plt.Rectangle((pc, rows - 1 - pr), 1, 1,
                                     facecolor='#fbbf24', alpha=0.45,
                                     linewidth=0)
                ax.add_patch(rect)

            # Draw path line
            xs = [pc + 0.5 for (pr, pc) in floor_path]
            ys = [rows - 1 - pr + 0.5 for (pr, pc) in floor_path]
            ax.plot(xs, ys, color='#f59e0b', linewidth=2.5, alpha=0.9,
                    marker='o', markersize=5, markerfacecolor='#fbbf24',
                    markeredgecolor='#f59e0b', markeredgewidth=1, zorder=10)

    # Draw start marker
    if start and start[0] == floor:
        sr, sc = start[1], start[2]
        ax.text(sc + 0.5, rows - 1 - sr + 0.5, '⭐',
                ha='center', va='center', fontsize=16, zorder=20)
        circle = plt.Circle((sc + 0.5, rows - 1 - sr + 0.5), 0.42,
                            fill=False, edgecolor='#fbbf24', linewidth=2.5,
                            linestyle='--', zorder=19)
        ax.add_patch(circle)

    # Draw goal marker
    if goal and goal[0] == floor:
        gr, gc = goal[1], goal[2]
        ax.text(gc + 0.5, rows - 1 - gr + 0.5, '🎯',
                ha='center', va='center', fontsize=16, zorder=20)
        circle = plt.Circle((gc + 0.5, rows - 1 - gr + 0.5), 0.42,
                            fill=False, edgecolor='#ef4444', linewidth=2.5,
                            linestyle='--', zorder=19)
        ax.add_patch(circle)

    # Axis styling
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels(range(cols), fontsize=8, color='#8888aa')
    ax.set_yticklabels(range(rows - 1, -1, -1), fontsize=8, color='#8888aa')
    ax.tick_params(axis='both', length=0)

    floor_name = parking_map.floor_names[floor] if floor < len(parking_map.floor_names) else f"Floor {floor + 1}"
    ax.set_title(f"🏢 {floor_name}", fontsize=14, fontweight='bold',
                 color='#e0e0ff', pad=12)

    # Legend
    legend_elements = []
    for ct_val, ct_name in CellType.NAMES.items():
        legend_elements.append(
            mpatches.Patch(facecolor=CellType.COLORS.get(ct_val, '#333'),
                           edgecolor='#444', label=ct_name)
        )
    if path:
        legend_elements.append(
            mpatches.Patch(facecolor='#fbbf24', alpha=0.6, label='Path')
        )
    if explored:
        legend_elements.append(
            mpatches.Patch(facecolor='#667eea', alpha=0.3, label='Explored')
        )

    legend = ax.legend(handles=legend_elements, loc='upper left',
                       bbox_to_anchor=(1.02, 1), frameon=True,
                       fontsize=8, facecolor='#1a1a35', edgecolor='#333366',
                       labelcolor='#c4c4e0')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    return fig


def render_metric_card(icon, label, value):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_recommendation_card(rank, rec):
    """Render a styled recommendation card."""
    reasons_html = ''.join(
        f'<div class="rec-reason">✦ {r}</div>' for r in rec.reasons
    )
    score_pct = min(rec.score, 100)
    return f"""
    <div class="rec-card">
        <div class="rec-title">#{rank} — Slot {rec.slot_id}</div>
        <div class="rec-score">
            Lantai {rec.floor + 1} &nbsp;•&nbsp; Posisi ({rec.row}, {rec.col})
            &nbsp;•&nbsp; Tipe: {rec.slot_type}
        </div>
        <div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
            <span style="color:#a78bfa; font-weight:700; font-size:1.1rem;">{score_pct:.1f}</span>
            <div class="score-bar-bg" style="flex:1;">
                <div class="score-bar-fill" style="width:{score_pct}%;"></div>
            </div>
        </div>
        {reasons_html}
    </div>
    """


def build_plotly_dark_layout(title):
    """Return a base Plotly layout dict for dark theme charts."""
    return dict(
        title=dict(text=title, font=dict(color='#e0e0ff', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,26,0.6)',
        font=dict(color='#a0a0c0'),
        xaxis=dict(gridcolor='rgba(100,100,180,0.15)', zerolinecolor='rgba(100,100,180,0.15)'),
        yaxis=dict(gridcolor='rgba(100,100,180,0.15)', zerolinecolor='rgba(100,100,180,0.15)'),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(font=dict(color='#c4c4e0')),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size: 3rem;">🅿️</div>
        <div style="font-size: 1.6rem; font-weight: 800;
             background: linear-gradient(135deg, #667eea, #a78bfa);
             -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">
            ParkNav AI
        </div>
        <div style="font-size: 0.8rem; color: #8888aa; margin-top: 4px;">
            Smart Parking Navigation System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Map selector
    st.markdown("#### 🗺️ Select Map")
    try:
        available_maps = get_available_maps()
    except Exception:
        available_maps = []
        st.warning("⚠️ No maps found. Ensure maps are in the `maps/` directory.")

    if available_maps:
        selected_map_name = st.selectbox("Map", available_maps, label_visibility="collapsed")
    else:
        selected_map_name = None
        st.error("No maps available.")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Navigation mode
    st.markdown("#### 🧭 Navigation Mode")
    nav_mode = st.radio(
        "Mode",
        ["🗺️ Manual Navigation", "🤖 Smart Recommendation", "📊 Algorithm Comparison"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Load parking map
    parking_map = None
    if selected_map_name:
        try:
            parking_map = load_map(selected_map_name)
        except Exception as e:
            st.error(f"❌ Error loading map: {e}")

    # Mode-specific controls
    start_pos = None
    goal_pos = None
    run_action = False
    vehicle_type = None
    preferences = {}
    recommendations = None

    if parking_map:
        if nav_mode == "🗺️ Manual Navigation":
            st.markdown("#### 📍 Start Position")
            entrance = parking_map.get_entrance()
            if entrance:
                st.info(f"Entrance: Floor {entrance[0]+1}, ({entrance[1]}, {entrance[2]})")
                start_pos = entrance
            else:
                s_floor = st.number_input("Floor", 0, parking_map.num_floors - 1, 0, key="sf")
                s_row = st.number_input("Row", 0, parking_map.num_rows - 1, 0, key="sr")
                s_col = st.number_input("Col", 0, parking_map.num_cols - 1, 0, key="sc")
                start_pos = (int(s_floor), int(s_row), int(s_col))

            st.markdown("#### 🎯 Goal Slot")
            avail_slots = parking_map.get_available_slots()
            if avail_slots:
                slot_ids = [s['id'] if isinstance(s, dict) else s for s in avail_slots]
                selected_slot = st.selectbox("Select slot", slot_ids)
                goal_pos = parking_map.get_slot_position(selected_slot)
                if goal_pos:
                    st.success(f"Slot {selected_slot}: Floor {goal_pos[0]+1}, ({goal_pos[1]}, {goal_pos[2]})")
            else:
                st.warning("No available slots on this map.")

            run_action = st.button("🚀 Find Route", use_container_width=True)

        elif nav_mode == "🤖 Smart Recommendation":
            st.markdown("#### 🚗 Vehicle Type")
            vehicle_type = st.selectbox("Vehicle", ["Sedan", "SUV", "Motor", "Disabilitas"])

            st.markdown("#### ⚙️ Preferences")
            pref_lower = st.checkbox("Prefer lower floor", True)
            pref_elevator = st.checkbox("Near elevator", False)
            pref_exit = st.checkbox("Near exit", True)
            pref_accessible = st.checkbox("Accessibility", vehicle_type == "Disabilitas")

            preferences = {
                "vehicle_type": vehicle_type.lower(),
                "prefer_lower_floor": pref_lower,
                "near_elevator": pref_elevator,
                "near_exit": pref_exit,
                "accessibility": pref_accessible,
            }

            run_action = st.button("🔍 Recommend", use_container_width=True)

        elif nav_mode == "📊 Algorithm Comparison":
            st.markdown("#### 📍 Start Position")
            entrance = parking_map.get_entrance()
            if entrance:
                st.info(f"Entrance: Floor {entrance[0]+1}, ({entrance[1]}, {entrance[2]})")
                start_pos = entrance
            else:
                s_floor = st.number_input("Floor", 0, parking_map.num_floors - 1, 0, key="csf")
                s_row = st.number_input("Row", 0, parking_map.num_rows - 1, 0, key="csr")
                s_col = st.number_input("Col", 0, parking_map.num_cols - 1, 0, key="csc")
                start_pos = (int(s_floor), int(s_row), int(s_col))

            st.markdown("#### 🎯 Goal Slot")
            avail_slots = parking_map.get_available_slots()
            if avail_slots:
                slot_ids = [s['id'] if isinstance(s, dict) else s for s in avail_slots]
                selected_slot = st.selectbox("Select slot", slot_ids, key="cslot")
                goal_pos = parking_map.get_slot_position(selected_slot)
                if goal_pos:
                    st.success(f"Slot {selected_slot}: Floor {goal_pos[0]+1}, ({goal_pos[1]}, {goal_pos[2]})")
            else:
                st.warning("No available slots.")

            run_action = st.button("📊 Compare Algorithms", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

if "nav_result" not in st.session_state:
    st.session_state.nav_result = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

# Execute actions
if run_action and parking_map:
    if nav_mode == "🗺️ Manual Navigation" and start_pos and goal_pos:
        try:
            result = astar_search(parking_map, start_pos, goal_pos)
            st.session_state.nav_result = result
            st.session_state.recommendations = None
            st.session_state.comparison_results = None
        except Exception as e:
            st.error(f"❌ Navigation error: {e}")

    elif nav_mode == "🤖 Smart Recommendation":
        try:
            expert = ParkingExpertSystem()
            recs = expert.recommend(parking_map, preferences)
            st.session_state.recommendations = recs
            st.session_state.nav_result = None
            st.session_state.comparison_results = None
        except Exception as e:
            st.error(f"❌ Recommendation error: {e}")

    elif nav_mode == "📊 Algorithm Comparison" and start_pos and goal_pos:
        try:
            algos = {
                "A*": astar_search,
                "BFS": bfs_search,
                "DFS": dfs_search,
                "Greedy": greedy_search,
            }
            comp_results = {}
            for name, fn in algos.items():
                comp_results[name] = fn(parking_map, start_pos, goal_pos)
            st.session_state.comparison_results = comp_results
            st.session_state.nav_result = None
            st.session_state.recommendations = None
        except Exception as e:
            st.error(f"❌ Comparison error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding: 10px 0 5px 0;">
    <h1 style="font-size: 2.4rem; font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;">
        🅿️ ParkNav AI
    </h1>
    <p style="color: #8888aa; font-size: 1rem; margin-top: 0;">
        Smart Parking Navigation with A* Search &amp; Expert System
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

if parking_map:
    tab_map, tab_results, tab_compare, tab_about = st.tabs([
        "🗺️ Parking Map", "📋 Results", "📊 Algorithm Comparison", "🧠 About A*"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: PARKING MAP
    # ─────────────────────────────────────────────────────────────────────────
    with tab_map:
        st.markdown('<div class="section-title">🏢 Parking Map Visualization</div>',
                    unsafe_allow_html=True)

        # Floor selector
        floor_options = []
        for i, fname in enumerate(parking_map.floor_names):
            floor_options.append(f"{fname}")
        if not floor_options:
            floor_options = [f"Floor {i+1}" for i in range(parking_map.num_floors)]

        selected_floor_label = st.radio(
            "Select Floor",
            floor_options,
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_floor_idx = floor_options.index(selected_floor_label)

        # Determine path & explored for this render
        render_path = None
        render_explored = None
        render_start = start_pos
        render_goal = goal_pos

        nav_result = st.session_state.nav_result
        if nav_result and nav_result.success:
            render_path = nav_result.path
            render_explored = nav_result.exploration_order

        # If comparison mode and we have A* result, show that path
        comp_results = st.session_state.comparison_results
        if comp_results and "A*" in comp_results and comp_results["A*"].success:
            render_path = comp_results["A*"].path
            render_explored = comp_results["A*"].exploration_order

        try:
            fig = render_parking_floor(
                parking_map, selected_floor_idx,
                path=render_path,
                explored=render_explored,
                start=render_start,
                goal=render_goal,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"❌ Error rendering map: {e}")

        # Map info
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.markdown(render_metric_card("🏢", "Floors", parking_map.num_floors),
                        unsafe_allow_html=True)
        with col_info2:
            st.markdown(render_metric_card("📐", "Grid Size",
                        f"{parking_map.num_rows}×{parking_map.num_cols}"),
                        unsafe_allow_html=True)
        with col_info3:
            try:
                total_avail = len(parking_map.get_available_slots())
            except Exception:
                total_avail = "?"
            st.markdown(render_metric_card("✅", "Available Slots", total_avail),
                        unsafe_allow_html=True)
        with col_info4:
            total_slots = len(parking_map.slots) if parking_map.slots else 0
            st.markdown(render_metric_card("🅿️", "Total Slots", total_slots),
                        unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_results:
        st.markdown('<div class="section-title">📋 Navigation & Recommendation Results</div>',
                    unsafe_allow_html=True)

        nav_result = st.session_state.nav_result
        recs = st.session_state.recommendations

        if nav_result:
            # Success or failure banner
            if nav_result.success:
                st.markdown(
                    f'<div class="success-banner">✅ Route Found! &nbsp;—&nbsp; {nav_result.algorithm} '
                    f'found a path with cost {nav_result.cost:.1f}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="error-banner">❌ No Route Found &nbsp;—&nbsp; {nav_result.message}</div>',
                    unsafe_allow_html=True,
                )

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_metric_card("💰", "Path Cost",
                            f"{nav_result.cost:.1f}"), unsafe_allow_html=True)
            with c2:
                st.markdown(render_metric_card("👣", "Steps",
                            len(nav_result.path)), unsafe_allow_html=True)
            with c3:
                st.markdown(render_metric_card("🔍", "Nodes Explored",
                            nav_result.nodes_explored), unsafe_allow_html=True)
            with c4:
                st.markdown(render_metric_card("⚡", "Time (ms)",
                            f"{nav_result.execution_time:.2f}"), unsafe_allow_html=True)

            # Step-by-step path
            if nav_result.success and nav_result.path:
                st.markdown("---")
                st.markdown('<div class="section-title">🗺️ Step-by-Step Path</div>',
                            unsafe_allow_html=True)

                # Group by floor
                floor_groups = {}
                for (f, r, c) in nav_result.path:
                    floor_groups.setdefault(f, []).append((r, c))

                for f_idx in sorted(floor_groups.keys()):
                    fname = (parking_map.floor_names[f_idx]
                             if f_idx < len(parking_map.floor_names) else f"Floor {f_idx+1}")
                    coords = floor_groups[f_idx]
                    coord_str = " → ".join(f"({r},{c})" for (r, c) in coords)
                    st.markdown(
                        f'<div class="path-step"><strong>🏢 {fname}:</strong> {coord_str}</div>',
                        unsafe_allow_html=True,
                    )

                # Additional info
                st.markdown("---")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(render_metric_card("🧩", "Nodes Generated",
                                nav_result.nodes_generated), unsafe_allow_html=True)
                with mc2:
                    st.markdown(render_metric_card("📊", "Algorithm",
                                nav_result.algorithm), unsafe_allow_html=True)

        elif recs:
            st.markdown(
                '<div class="success-banner">🤖 Expert System Recommendations Ready!</div>',
                unsafe_allow_html=True,
            )

            if not recs:
                st.warning("No recommendations available for the given preferences.")
            else:
                top_n = min(5, len(recs))
                st.markdown(f"**Top {top_n} Recommended Slots:**")
                for idx, rec in enumerate(recs[:top_n], 1):
                    st.markdown(render_recommendation_card(idx, rec),
                                unsafe_allow_html=True)

                # Quick-navigate option
                st.markdown("---")
                st.markdown('<div class="section-title">🚀 Quick Navigate to Recommendation</div>',
                            unsafe_allow_html=True)
                st.info("💡 Switch to **Manual Navigation** mode and select the recommended "
                        "slot to find the optimal route.")

        else:
            st.markdown("""
            <div style="text-align:center; padding: 60px 20px; color: #6666aa;">
                <div style="font-size: 4rem; margin-bottom: 16px;">🧭</div>
                <div style="font-size: 1.2rem; font-weight: 600;">No results yet</div>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    Select a mode from the sidebar, configure your options, and click the action button.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: ALGORITHM COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    with tab_compare:
        st.markdown('<div class="section-title">📊 Algorithm Performance Comparison</div>',
                    unsafe_allow_html=True)

        comp_results = st.session_state.comparison_results

        if comp_results:
            # ── Comparison table ──
            st.markdown("#### 📋 Comparison Table")

            table_data = {
                "Metric": [
                    "Success", "Path Cost", "Steps", "Nodes Explored",
                    "Nodes Generated", "Time (ms)", "Optimal?",
                ],
            }
            optimal_props = {"A*": "✅", "BFS": "✅*", "DFS": "❌", "Greedy": "❌"}

            for algo_name, res in comp_results.items():
                table_data[algo_name] = [
                    "✅" if res.success else "❌",
                    f"{res.cost:.1f}" if res.success else "—",
                    str(len(res.path)) if res.success else "—",
                    str(res.nodes_explored),
                    str(res.nodes_generated),
                    f"{res.execution_time:.2f}",
                    optimal_props.get(algo_name, "—"),
                ]

            st.table(table_data)
            st.caption("*BFS optimal hanya untuk graf dengan bobot seragam (unweighted).")

            st.markdown("---")

            # ── Charts ──
            algo_names = list(comp_results.keys())
            algo_colors = ['#6366f1', '#3b82f6', '#f59e0b', '#10b981']

            # Chart 1: Path Cost
            costs = [comp_results[a].cost if comp_results[a].success else 0
                     for a in algo_names]
            fig_cost = go.Figure(data=[
                go.Bar(x=algo_names, y=costs,
                       marker=dict(color=algo_colors,
                                   line=dict(color='rgba(255,255,255,0.1)', width=1)),
                       text=[f"{c:.1f}" for c in costs], textposition='outside',
                       textfont=dict(color='#e0e0ff'))
            ])
            fig_cost.update_layout(**build_plotly_dark_layout("💰 Path Cost Comparison"))
            st.plotly_chart(fig_cost, use_container_width=True)

            # Chart 2: Nodes Explored
            nodes = [comp_results[a].nodes_explored for a in algo_names]
            fig_nodes = go.Figure(data=[
                go.Bar(x=algo_names, y=nodes,
                       marker=dict(color=algo_colors,
                                   line=dict(color='rgba(255,255,255,0.1)', width=1)),
                       text=[str(n) for n in nodes], textposition='outside',
                       textfont=dict(color='#e0e0ff'))
            ])
            fig_nodes.update_layout(**build_plotly_dark_layout("🔍 Nodes Explored Comparison"))
            st.plotly_chart(fig_nodes, use_container_width=True)

            # Chart 3: Execution Time
            times = [comp_results[a].execution_time for a in algo_names]
            fig_time = go.Figure(data=[
                go.Bar(x=algo_names, y=times,
                       marker=dict(color=algo_colors,
                                   line=dict(color='rgba(255,255,255,0.1)', width=1)),
                       text=[f"{t:.2f}" for t in times], textposition='outside',
                       textfont=dict(color='#e0e0ff'))
            ])
            fig_time.update_layout(**build_plotly_dark_layout("⚡ Execution Time (ms) Comparison"))
            st.plotly_chart(fig_time, use_container_width=True)

            # Radar chart
            st.markdown("---")
            st.markdown("#### 🕸️ Overall Performance Radar")

            # Normalize metrics to 0-1 for radar
            max_cost = max(costs) if max(costs) > 0 else 1
            max_nodes = max(nodes) if max(nodes) > 0 else 1
            max_time = max(times) if max(times) > 0 else 1

            radar_fig = go.Figure()
            categories = ['Path Optimality', 'Node Efficiency', 'Speed', 'Completeness', 'Consistency']

            for idx, algo in enumerate(algo_names):
                res = comp_results[algo]
                # Higher is better for radar, so invert cost and nodes
                cost_score = 1 - (res.cost / max_cost) if res.success else 0
                node_score = 1 - (res.nodes_explored / max_nodes)
                time_score = 1 - (res.execution_time / max_time)
                complete_score = 1.0 if res.success else 0.0
                consistent_score = 1.0 if algo in ("A*", "BFS") else 0.5

                values = [cost_score, node_score, time_score, complete_score, consistent_score]
                values.append(values[0])  # close the polygon

                radar_fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=algo,
                    line=dict(color=algo_colors[idx], width=2),
                    fillcolor='rgba({},{},{},0.1)'.format(
                        int(algo_colors[idx][1:3], 16),
                        int(algo_colors[idx][3:5], 16),
                        int(algo_colors[idx][5:7], 16),
                    ) if algo_colors[idx].startswith('#')
                    else algo_colors[idx].replace(')', ',0.1)').replace('rgb', 'rgba'),
                ))

            radar_fig.update_layout(
                polar=dict(
                    bgcolor='rgba(15,15,26,0.6)',
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor='rgba(100,100,180,0.15)',
                                    color='#8888aa'),
                    angularaxis=dict(gridcolor='rgba(100,100,180,0.15)',
                                     color='#c4c4e0'),
                ),
                **{k: v for k, v in build_plotly_dark_layout("🕸️ Performance Radar").items()
                   if k not in ('xaxis', 'yaxis')},
                showlegend=True,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding: 60px 20px; color: #6666aa;">
                <div style="font-size: 4rem; margin-bottom: 16px;">📊</div>
                <div style="font-size: 1.2rem; font-weight: 600;">No comparison data</div>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    Switch to <strong>📊 Algorithm Comparison</strong> mode, select start &amp; goal, and click <strong>Compare</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: ABOUT A*
    # ─────────────────────────────────────────────────────────────────────────
    with tab_about:
        st.markdown('<div class="section-title">🧠 About A* Search Algorithm</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="about-block">
            <h3>📌 Apa itu A* Search?</h3>
            <p>
                <strong>A* (A-star)</strong> adalah algoritma pencarian informed yang menemukan jalur optimal
                dari titik awal ke titik tujuan pada graf berbobot. Algoritma ini menggabungkan dua komponen:
            </p>
            <ul>
                <li><code>g(n)</code> — Biaya aktual dari node awal ke node <em>n</em></li>
                <li><code>h(n)</code> — Estimasi heuristik biaya dari node <em>n</em> ke tujuan</li>
                <li><code>f(n) = g(n) + h(n)</code> — Fungsi evaluasi total</li>
            </ul>
            <p>
                A* selalu mengekspansi node dengan nilai <code>f(n)</code> terkecil terlebih dahulu,
                sehingga menjamin efisiensi dalam pencarian.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-block">
            <h3>✅ Pembuktian Admissibility</h3>
            <p>
                Heuristik <code>h(n)</code> bersifat <strong>admissible</strong> jika tidak pernah
                meng-<em>overestimate</em> biaya sebenarnya untuk mencapai tujuan:
            </p>
            <p style="text-align:center; font-size:1.1rem; color:#a78bfa; margin:12px 0;">
                <code>h(n) ≤ h*(n)</code> untuk semua node <em>n</em>
            </p>
            <p>
                <strong>Pada sistem ini:</strong> Heuristik yang digunakan adalah 3D Manhattan Distance:
            </p>
            <p style="text-align:center; color:#a78bfa;">
                <code>h(n) = |x₁ - x₂| + |y₁ - y₂| + |floor₁ - floor₂| × 1.0</code>
            </p>
            <p>
                Karena <code>floor_weight = 1.0</code> lebih kecil dari biaya minimum perpindahan
                antar lantai via ramp (<code>1.5</code>), maka heuristik ini <strong>tidak pernah
                overestimate</strong> → <strong>Admissible</strong> ✅
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-block">
            <h3>🔗 Pembuktian Consistency (Monotonicity)</h3>
            <p>
                Heuristik <code>h(n)</code> bersifat <strong>consistent</strong> jika memenuhi
                <em>triangle inequality</em>:
            </p>
            <p style="text-align:center; font-size:1.1rem; color:#a78bfa; margin:12px 0;">
                <code>h(n) ≤ c(n, n') + h(n')</code>
            </p>
            <p>untuk setiap edge <code>(n, n')</code> dengan cost <code>c(n, n')</code>.</p>
            <ul>
                <li>
                    <strong>Perpindahan horizontal/vertikal:</strong>
                    <code>|h(n) - h(n')| ≤ 1 ≤ c(n,n') = 1.0</code> ✅
                </li>
                <li>
                    <strong>Perpindahan antar lantai (ramp):</strong>
                    <code>|h(n) - h(n')| ≤ 1.0 ≤ c(n,n') = 1.5</code> ✅
                </li>
            </ul>
            <p>
                Karena consistency terpenuhi, maka heuristik juga <strong>admissible</strong>,
                dan A* dijamin <strong>tidak perlu re-expand</strong> node yang sudah dikunjungi.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-block">
            <h3>🏆 Jaminan A*</h3>
            <table style="width:100%; border-collapse:collapse; margin:12px 0;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">Optimal</td>
                    <td style="padding:8px;">✅ Selalu menemukan rute dengan biaya minimum</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">Complete</td>
                    <td style="padding:8px;">✅ Selalu menemukan solusi jika ada jalur yang tersedia</td>
                </tr>
                <tr>
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">Efficient</td>
                    <td style="padding:8px;">✅ Mengeksplorasi node lebih sedikit daripada BFS berkat heuristik</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-block">
            <h3>🤖 Rule-Based Expert System</h3>
            <p>
                Sistem pakar berbasis aturan digunakan untuk merekomendasikan slot parkir
                berdasarkan preferensi pengguna. Setiap aturan memberikan skor pada slot yang tersedia:
            </p>
            <table style="width:100%; border-collapse:collapse; margin:12px 0;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">1. Vehicle Size</td>
                    <td style="padding:8px;">Cocokkan ukuran slot dengan jenis kendaraan</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">2. Floor Preference</td>
                    <td style="padding:8px;">Prioritaskan lantai bawah untuk akses cepat</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">3. Elevator Proximity</td>
                    <td style="padding:8px;">Dekatkan slot ke elevator untuk kenyamanan</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">4. Exit Proximity</td>
                    <td style="padding:8px;">Dekatkan slot ke pintu keluar</td>
                </tr>
                <tr>
                    <td style="padding:8px; color:#a78bfa; font-weight:700;">5. Accessibility</td>
                    <td style="padding:8px;">Prioritaskan slot aksesibel untuk disabilitas</td>
                </tr>
            </table>
            <p>
                Slot dengan total skor tertinggi akan direkomendasikan kepada pengguna,
                beserta alasan dari setiap aturan yang berkontribusi.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-block">
            <h3>📊 Perbandingan Algoritma</h3>
            <table style="width:100%; border-collapse:collapse; margin:12px 0;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.15); color:#a78bfa;">
                    <th style="padding:10px; text-align:left;">Algoritma</th>
                    <th style="padding:10px; text-align:center;">Optimal</th>
                    <th style="padding:10px; text-align:center;">Complete</th>
                    <th style="padding:10px; text-align:center;">Informed</th>
                    <th style="padding:10px; text-align:left;">Keterangan</th>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px; font-weight:600;">A* Search</td>
                    <td style="padding:10px; text-align:center;">✅</td>
                    <td style="padding:10px; text-align:center;">✅</td>
                    <td style="padding:10px; text-align:center;">✅</td>
                    <td style="padding:10px;">Terbaik — menggunakan g(n) + h(n)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px; font-weight:600;">BFS</td>
                    <td style="padding:10px; text-align:center;">✅*</td>
                    <td style="padding:10px; text-align:center;">✅</td>
                    <td style="padding:10px; text-align:center;">❌</td>
                    <td style="padding:10px;">Optimal hanya pada unweighted graph</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:10px; font-weight:600;">DFS</td>
                    <td style="padding:10px; text-align:center;">❌</td>
                    <td style="padding:10px; text-align:center;">❌</td>
                    <td style="padding:10px; text-align:center;">❌</td>
                    <td style="padding:10px;">Bisa terjebak, tidak optimal</td>
                </tr>
                <tr>
                    <td style="padding:10px; font-weight:600;">Greedy</td>
                    <td style="padding:10px; text-align:center;">❌</td>
                    <td style="padding:10px; text-align:center;">✅</td>
                    <td style="padding:10px; text-align:center;">✅</td>
                    <td style="padding:10px;">Cepat tapi hanya menggunakan h(n)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

else:
    # No parking map loaded
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color: #6666aa;">
        <div style="font-size: 5rem; margin-bottom: 20px;">🅿️</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #8888cc;">
            Welcome to ParkNav AI
        </div>
        <div style="font-size: 1rem; margin-top: 12px; max-width: 500px; margin-left: auto; margin-right: auto;">
            Select a parking map from the sidebar to get started.
            <br>Ensure map files are available in the <code>maps/</code> directory.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 16px 0; color: #555577; font-size: 0.8rem;">
    🅿️ <strong>ParkNav AI</strong> — Smart Parking Navigation System
    <br>Built with Streamlit • A* Search • Expert System
    <br>UAS Kecerdasan Buatan &copy; 2024
</div>
""", unsafe_allow_html=True)
