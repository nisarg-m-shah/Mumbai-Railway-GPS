"""
Mumbai Multimodal Route Planner — Streamlit Frontend
=====================================================
Run with:   streamlit run app.py

Expects the following files in the same directory:
  - GPS.py
  - mumbai_graph.graphml
  - Cleaned Local Train Dataset.csv
  - railway_mumbai.csv
  - Bus Dataset 3.csv
"""

import io
import sys
import os

# Ensure the directory containing this file is on the path so that
# mumbai_multimodal_planner.py is always importable regardless of the
# working directory Streamlit uses when launching the app.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import osmnx as ox

# ── import the engine ─────────────────────────────────────────────────────────
from GPS import (
    MumbaiData, Router, normalize,
    WALK_MIN_PER_KM, BUS_MIN_PER_KM, CAR_MIN_PER_KM,
    MODE_COLORS,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai Multimodal Planner",
    page_icon="🚇",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADING  (only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Mumbai transport data…")
def load_data():
    return MumbaiData()

@st.cache_resource(show_spinner=False)
def get_router(_data):
    return Router(_data)

# ──────────────────────────────────────────────────────────────────────────────
# ROUTE DETAIL FORMATTER
# ──────────────────────────────────────────────────────────────────────────────
MODE_EMOJI = {"walk": "🚶", "train": "🚆", "bus": "🚌", "car": "🚖"}

def friendly_node(node_id: str, data: MumbaiData) -> str:
    """Turn internal node IDs into human-readable labels."""
    if node_id == "start":
        return "📍 Start"
    if node_id == "end":
        return "🏁 Destination"
    if node_id.startswith("train_"):
        sc = node_id[len("train_"):]
        return data.station_display.get(sc, sc).title()
    if node_id.startswith("bus_"):
        stop_id = node_id[len("bus_"):]
        # Look up stop name from bus_df
        rows = data.bus_df[data.bus_df["stop_id"] == stop_id]["stop_name"]
        return rows.iloc[0] if not rows.empty else stop_id
    return node_id


def render_route_steps(steps, data: MumbaiData):
    """Render a detailed step-by-step breakdown with mode-specific info."""

    # ── group consecutive steps of the same mode+route into legs ──────────
    legs = []
    for step in steps:
        mode  = step["mode"]
        route = step.get("route", "")
        if legs and legs[-1]["mode"] == mode and legs[-1]["route"] == route:
            legs[-1]["steps"].append(step)
            legs[-1]["distance_km"] += step["distance_km"]
            legs[-1]["time_min"]    += step["time_min"]
        else:
            legs.append({
                "mode":        mode,
                "route":       route,
                "steps":       [step],
                "distance_km": step["distance_km"],
                "time_min":    step["time_min"],
            })

    for i, leg in enumerate(legs):
        mode     = leg["mode"]
        emoji    = MODE_EMOJI.get(mode, "•")
        color    = MODE_COLORS.get(mode, "black")
        first_step = leg["steps"][0]
        last_step  = leg["steps"][-1]

        from_label = friendly_node(first_step["from"], data)
        to_label   = friendly_node(last_step["to"],   data)

        # ── header ──
        if mode == "train":
            line = leg["route"]
            header = f"{emoji} **Train** — {line} line"
        elif mode == "bus":
            header = f"{emoji} **Bus** — Route {leg['route']}"
        elif mode == "walk":
            header = f"{emoji} **Walk**"
        else:
            header = f"{emoji} **Cab**"

        with st.expander(
            f"Leg {i+1}: {header}  |  "
            f"{leg['distance_km']:.2f} km  |  "
            f"{leg['time_min']:.1f} min",
            expanded=True,
        ):
            st.markdown(f"**From:** {from_label}  →  **To:** {to_label}")

            if mode == "train":
                # List every station on this leg
                stops_passed = []
                for s in leg["steps"]:
                    if not stops_passed:
                        stops_passed.append(friendly_node(s["from"], data))
                    stops_passed.append(friendly_node(s["to"], data))

                # Detect line changes within the leg (shouldn't normally happen,
                # but handle gracefully)
                lines_used = list(dict.fromkeys(
                    s.get("route","") for s in leg["steps"]
                ))
                if len(lines_used) > 1:
                    st.info(f"Line change: {' → '.join(lines_used)}")

                st.markdown("**Stations passed:**")
                cols = st.columns(3)
                for j, sname in enumerate(stops_passed):
                    cols[j % 3].markdown(f"• {sname}")

            elif mode == "bus":
                route_no = leg["route"]
                st.markdown(f"**Bus route:** {route_no}")
                stops_passed = []
                for s in leg["steps"]:
                    if not stops_passed:
                        stops_passed.append(friendly_node(s["from"], data))
                    stops_passed.append(friendly_node(s["to"], data))

                st.markdown("**Stops passed:**")
                cols = st.columns(3)
                for j, sname in enumerate(stops_passed):
                    cols[j % 3].markdown(f"• {sname}")

            elif mode == "walk":
                if leg["distance_km"] >= 1.0:
                    st.warning(
                        f"Walk distance {leg['distance_km']:.2f} km exceeds 1 km. "
                        "Consider taking a cab for this leg."
                    )

            st.caption(
                f"Distance: {leg['distance_km']:.2f} km  |  "
                f"Time: {leg['time_min']:.1f} min"
            )


# ──────────────────────────────────────────────────────────────────────────────
# MAP RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def render_map(steps, start_coords, end_coords, G_road) -> io.BytesIO:
    """
    Render the route on the OSMnx road graph.
    - Crops the view to the route bounding box + padding (no more full-Mumbai view)
    - Train: straight lines between station coords
    - Walk/Bus/Cab: actual road geometry via OSMnx shortest path
    - Distinct colors: Walk=gray, Train=red, Bus=dodgerblue, Cab=gold
    """

    # Collect all coordinates in the route for bbox calculation
    all_lats = [start_coords[0], end_coords[0]]
    all_lons = [start_coords[1], end_coords[1]]
    for step in steps:
        for key in ("seg_start", "seg_end"):
            coord = step.get(key)
            if coord:
                all_lats.append(coord[0])
                all_lons.append(coord[1])

    PAD = 0.02   # degrees of padding around the route
    min_lat, max_lat = min(all_lats) - PAD, max(all_lats) + PAD
    min_lon, max_lon = min(all_lons) - PAD, max(all_lons) + PAD

    # Cab is gold/yellow so it does not clash with the lime Start marker
    LOCAL_COLORS = {
        "walk":  "#aaaaaa",
        "train": "#ff3333",
        "bus":   "#1e90ff",
        "car":   "#ffd700",
    }

    fig, ax = ox.plot_graph(
        G_road,
        show=False, close=False,
        node_size=0,
        edge_color="#3a3a5c",
        edge_linewidth=0.4,
        bgcolor="#1a1a2e",
        figsize=(12, 12),
    )

    for step in steps:
        mode  = step["mode"]
        color = LOCAL_COLORS.get(mode, "white")
        seg_s = step.get("seg_start")
        seg_e = step.get("seg_end")
        if seg_s is None or seg_e is None:
            continue

        if mode == "train":
            ax.plot([seg_s[1], seg_e[1]], [seg_s[0], seg_e[0]],
                    color=color, linewidth=4, alpha=0.95,
                    solid_capstyle="round", zorder=5)
            ax.scatter(seg_s[1], seg_s[0], c=color, s=50, zorder=6)
            ax.scatter(seg_e[1], seg_e[0], c=color, s=50, zorder=6)
        else:
            try:
                n1 = ox.distance.nearest_nodes(G_road, seg_s[1], seg_s[0])
                n2 = ox.distance.nearest_nodes(G_road, seg_e[1], seg_e[0])
                road_path = nx.shortest_path(G_road, n1, n2, weight="length")
                lons = [G_road.nodes[n]["x"] for n in road_path]
                lats = [G_road.nodes[n]["y"] for n in road_path]
                lw = 4 if mode == "bus" else 3
                ax.plot(lons, lats, color=color, linewidth=lw,
                        alpha=0.95, solid_capstyle="round", zorder=5)
            except Exception:
                ax.plot([seg_s[1], seg_e[1]], [seg_s[0], seg_e[0]],
                        color=color, linewidth=2, linestyle="--",
                        alpha=0.8, zorder=4)

    ax.scatter(start_coords[1], start_coords[0],
               c="lime", s=400, zorder=10, marker="*")
    ax.scatter(end_coords[1], end_coords[0],
               c="orange", s=400, zorder=10, marker="*")

    # Crop to route bounding box
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    patches = [mpatches.Patch(color=c, label=m.title())
               for m, c in LOCAL_COLORS.items()]
    patches += [mpatches.Patch(color="lime",   label="Start"),
                mpatches.Patch(color="orange", label="End")]
    ax.legend(handles=patches, loc="upper left",
              fontsize=9, facecolor="#2a2a3e", labelcolor="white",
              framealpha=0.85)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🚇 Mumbai Multimodal Route Planner")
    st.caption("Train · Bus · Cab · Walking — shortest-path across all modes")

    # ── Load data ─────────────────────────────────────────────────────────
    try:
        data   = load_data()
        router = get_router(data)
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}")
        st.stop()

    # ── Sidebar inputs ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Route Settings")

        st.subheader("📍 Start Location")
        start_lat = st.number_input("Start Latitude",  value=19.0760, format="%.6f", step=0.0001)
        start_lon = st.number_input("Start Longitude", value=72.8777, format="%.6f", step=0.0001)

        st.subheader("🏁 Destination")
        end_lat = st.number_input("Destination Latitude",  value=19.1972, format="%.6f", step=0.0001)
        end_lon = st.number_input("Destination Longitude", value=72.9780, format="%.6f", step=0.0001)

        st.subheader("🚦 Travel Mode")
        mode = st.selectbox(
            "Mode",
            options=["train", "bus", "car", "combination"],
            format_func=lambda m: {
                "train":       "🚆 Train only",
                "bus":         "🚌 Bus only",
                "car":         "🚖 Cab only",
                "combination": "🔀 Combination (all modes)",
            }[m],
        )

        compute = st.button("🔍 Find Route", type="primary", use_container_width=True)

        st.divider()
        st.caption(
            "**Speed model**\n"
            "- Walk: 12 min/km\n"
            "- Bus: 4 min/km\n"
            "- Train: timetable data\n"
            "- Cab: 3 min/km *(min 1 km)*"
        )

    # ── Route computation ─────────────────────────────────────────────────
    if not compute:
        st.info("Set your start and destination coordinates in the sidebar, then click **Find Route**.")
        return

    start_coords = (start_lat, start_lon)
    end_coords   = (end_lat,   end_lon)

    with st.spinner(f"Computing {mode} route…"):
        try:
            result = router.route(start_coords, end_coords, mode)
        except Exception as e:
            st.error(f"Routing error: {e}")
            return

    if result[0] is None:
        st.error("No route found between the selected locations.")
        if result[3]:
            for a in result[3]:
                st.warning(a)
        return

    path, steps, total_time, advisories, G_multi = result

    # ── Summary metrics ───────────────────────────────────────────────────
    total_dist = sum(s["distance_km"] for s in steps)
    mode_times = {}
    for s in steps:
        mode_times[s["mode"]] = mode_times.get(s["mode"], 0) + s["time_min"]

    st.success(f"Route found! Total time: **{total_time:.1f} min** | Distance: **{total_dist:.2f} km**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱ Total Time",   f"{total_time:.1f} min")
    col2.metric("📏 Total Dist",   f"{total_dist:.2f} km")
    col3.metric("🔢 Legs",         len([l for l in steps]))
    col4.metric("🔄 Mode changes", len(set(s["mode"] for s in steps)))

    # ── Mode time breakdown ───────────────────────────────────────────────
    if len(mode_times) > 1:
        st.subheader("Time breakdown by mode")
        bcols = st.columns(len(mode_times))
        for i, (m, t) in enumerate(mode_times.items()):
            bcols[i].metric(
                f"{MODE_EMOJI.get(m, '')} {m.title()}",
                f"{t:.1f} min",
                f"{t/total_time*100:.0f}%"
            )

    # ── Advisories ────────────────────────────────────────────────────────
    if advisories:
        with st.expander("⚠️ Advisories", expanded=False):
            for a in advisories:
                st.warning(a)

    st.divider()

    # ── Two-column layout: steps | map ───────────────────────────────────
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("📋 Route Steps")
        render_route_steps(steps, data)

    with right:
        st.subheader("🗺️ Route Map")
        with st.spinner("Rendering map…"):
            try:
                buf = render_map(steps, start_coords, end_coords, data.G_road)
                st.image(buf, use_container_width=True,
                         caption="Gray=Walk · Red=Train · Blue=Bus · Gold=Cab")
            except Exception as e:
                st.error(f"Map rendering failed: {e}")


if __name__ == "__main__":
    main()