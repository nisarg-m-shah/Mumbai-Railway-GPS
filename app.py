"""
Mumbai Multimodal Route Planner — Streamlit Frontend
=====================================================
Run with:   streamlit run app.py
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import osmnx as ox

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
# THEME / GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Nunito:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #0a1628;
    color: #e8eaf6;
}
h1, h2, h3, .metric-label { font-family: 'Space Grotesk', sans-serif; }

/* hero header */
.hero {
    background: linear-gradient(135deg, #00e5c0 0%, #1e90ff 50%, #9c27b0 100%);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    border-radius: 16px;
    padding: 28px 32px 20px;
    margin-bottom: 24px;
    color: #fff;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero h1 { font-size: 2.2rem; margin: 0 0 6px; }
.hero p  { margin: 0; opacity: 0.9; font-size: 1.05rem; }
.mode-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 3px 12px;
    margin: 4px 4px 0 0;
    font-size: 0.85rem;
    font-weight: 600;
}

/* feature cards */
.feature-card {
    background: linear-gradient(145deg, #1a2540, #0f1d35);
    border: 1px solid #2a3a60;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: transform 0.2s;
}
.feature-card:hover { transform: translateY(-3px); }
.feature-icon { font-size: 2rem; margin-bottom: 8px; }
.feature-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; margin: 0 0 6px; }
.feature-desc  { font-size: 0.9rem; color: #9aa7c4; margin: 0; }

/* success banner */
.success-banner {
    background: linear-gradient(90deg, #00897b, #1565c0);
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
}
.success-banner .big { font-size: 1.5rem; font-weight: 700; }
.success-banner .small { font-size: 0.9rem; opacity: 0.85; }

/* leg cards */
.leg-card {
    background: #111d35;
    border-radius: 10px;
    border-left: 5px solid #ccc;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.leg-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.leg-route   { color: #9aa7c4; font-size: 0.85rem; }
.leg-stats   { font-size: 0.8rem; color: #667; background: #1c2d4a; border-radius: 6px; padding: 2px 8px; }
.leg-body    { font-size: 0.88rem; color: #b0bec5; margin-top: 8px; }
.leg-points  { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.stop-chip {
    display: inline-block;
    border-radius: 12px;
    padding: 2px 9px;
    font-size: 0.78rem;
    margin: 2px;
    font-weight: 600;
}
.walk-warn {
    background: #2d1f00;
    border: 1px solid #f57c00;
    border-radius: 6px;
    padding: 6px 10px;
    color: #ffb74d;
    font-size: 0.82rem;
    margin-top: 6px;
}
.progress-bar-wrap { border-radius: 8px; overflow: hidden; display: flex; height: 14px; margin: 8px 0 16px; }
.progress-segment { height: 100%; }

/* sidebar card */
.sidebar-card {
    background: #121f38;
    border: 1px solid #1e3060;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

/* map legend */
.map-legend {
    display: flex; flex-wrap: wrap; gap: 10px;
    padding: 8px 12px;
    background: #0e1a30;
    border-radius: 8px;
    margin-top: 8px;
    font-size: 0.82rem;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.legend-dot  { width: 12px; height: 12px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Mumbai transport data…")
def load_data():
    return MumbaiData()

@st.cache_resource(show_spinner=False)
def get_router(_data):
    return Router(_data)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
MODE_EMOJI  = {"walk": "🚶", "train": "🚆", "bus": "🚌", "car": "🚖"}
MODE_HEX    = {
    "train": "#ff3333",
    "bus":   "#1e90ff",
    "walk":  "#aaaaaa",
    "car":   "#ffd700",
}

def friendly_node(node_id: str, data: MumbaiData) -> str:
    """
    Convert internal node IDs to human-readable labels.
    train_<Station Name>__<Line>  →  Station Name  (strip the __Line suffix)
    bus_<stop_id>                 →  stop_name from bus_df
    start / end                   →  icons
    """
    if node_id == "start":
        return "📍 Start"
    if node_id == "end":
        return "🏁 Destination"
    if node_id.startswith("train_"):
        # node_id = "train_Andheri__Western"
        inner = node_id[len("train_"):]          # "Andheri__Western"
        station_name = inner.split("__")[0]      # "Andheri"
        return station_name.title()
    if node_id.startswith("bus_"):
        stop_id = node_id[len("bus_"):]
        rows = data.bus_df[data.bus_df["stop_id"] == stop_id]["stop_name"]
        return rows.iloc[0].title() if not rows.empty else stop_id
    return node_id


def group_into_legs(steps):
    """Merge consecutive steps with the same mode+route into a single leg."""
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
    return legs


def legs_for_display(legs):
    """Filter out walk legs under 100 m (but their time still counts in total)."""
    return [l for l in legs if not (l["mode"] == "walk" and l["distance_km"] < 0.1)]


def render_progress_bar(legs, total_time):
    segments = ""
    for leg in legs:
        pct   = (leg["time_min"] / total_time) * 100 if total_time else 0
        color = MODE_HEX.get(leg["mode"], "#888")
        title = f"{leg['mode'].title()}: {leg['time_min']:.1f} min"
        segments += (f'<div class="progress-segment" '
                     f'style="width:{pct:.1f}%; background:{color};" title="{title}"></div>')
    st.markdown(f'<div class="progress-bar-wrap">{segments}</div>', unsafe_allow_html=True)


def render_leg_card(leg, idx, data: MumbaiData):
    mode  = leg["mode"]
    color = MODE_HEX.get(mode, "#888")
    emoji = MODE_EMOJI.get(mode, "•")

    first_step = leg["steps"][0]
    last_step  = leg["steps"][-1]
    from_label = friendly_node(first_step["from"], data)
    to_label   = friendly_node(last_step["to"],   data)

    # header line
    if mode == "train":
        route_label = f"<span class='leg-route'>🛤 {leg['route']}</span>"
        header_text = f"{emoji} Train"
    elif mode == "bus":
        route_label = f"<span class='leg-route'>🔢 Route {leg['route']}</span>"
        header_text = f"{emoji} Bus"
    elif mode == "walk":
        route_label = ""
        header_text = f"{emoji} Walk"
    else:
        route_label = ""
        header_text = f"{emoji} Cab"

    stats = (f"<span class='leg-stats'>"
             f"📏 {leg['distance_km']:.2f} km &nbsp;|&nbsp; "
             f"⏱ {leg['time_min']:.1f} min</span>")

    # stop chips
    stop_chip_color = color + "33"   # 20% opacity background
    stops_passed = []
    for s in leg["steps"]:
        fn = friendly_node(s["from"], data)
        if not stops_passed:
            stops_passed.append(fn)
        stops_passed.append(friendly_node(s["to"], data))

    chips = ""
    for i, sname in enumerate(stops_passed):
        if i > 0 and i < len(stops_passed) - 1 and mode in ("train", "bus") and len(stops_passed) > 5:
            if i == 1:
                chips += f"<span style='color:{color};font-size:0.8rem;'>…{len(stops_passed)-2} stops…</span>"
            continue
        chips += (f"<span class='stop-chip' "
                  f"style='background:{stop_chip_color};color:{color};border:1px solid {color}55;'>"
                  f"{sname}</span>")

    walk_warn = ""
    if mode == "walk" and leg["distance_km"] >= 1.0:
        walk_warn = (f"<div class='walk-warn'>"
                     f"⚠️ Walk distance {leg['distance_km']:.2f} km — consider a cab for this leg."
                     f"</div>")

    card_html = f"""
<div class="leg-card" style="border-left-color:{color};">
  <div class="leg-header">
    <span>{header_text} &nbsp; {route_label}</span>
    {stats}
  </div>
  <div class="leg-body">
    <div class="leg-points">
      <strong style="color:{color};">{from_label}</strong>
      <span style="color:#667;">→</span>
      <strong style="color:{color};">{to_label}</strong>
    </div>
    <div style="margin-top:6px;">{chips}</div>
    {walk_warn}
  </div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAP RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def render_map(steps, start_coords, end_coords, G_road, data) -> io.BytesIO:
    all_lats = [start_coords[0], end_coords[0]]
    all_lons = [start_coords[1], end_coords[1]]
    for step in steps:
        for key in ("seg_start", "seg_end"):
            coord = step.get(key)
            if coord:
                all_lats.append(coord[0])
                all_lons.append(coord[1])

    PAD = 0.02
    min_lat, max_lat = min(all_lats) - PAD, max(all_lats) + PAD
    min_lon, max_lon = min(all_lons) - PAD, max(all_lons) + PAD

    LOCAL_COLORS = {
        "walk":  "#aaaaaa",
        "train": "#ff3333",
        "bus":   "#1e90ff",
        "car":   "#ffd700",
    }

    fig, ax = ox.plot_graph(
        G_road, show=False, close=False,
        node_size=0, edge_color="#3a3a5c",
        edge_linewidth=0.4, bgcolor="#0a1628",
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
                        color=color, linewidth=2, linestyle="--", alpha=0.8, zorder=4)

    # ── Stop labels ──
    stop_labels = {}
    for step in steps:
        if step["mode"] in ("train", "bus"):
            for coord_key, node_key in [("seg_start", "from"), ("seg_end", "to")]:
                coord = step.get(coord_key)
                node  = step.get(node_key)
                if coord and coord not in stop_labels:
                    stop_labels[coord] = friendly_node(node, data)

    for (lat, lon), label in stop_labels.items():
        is_train = any(
            s["mode"] == "train" and (
                s.get("seg_start") == (lat, lon) or s.get("seg_end") == (lat, lon))
            for s in steps
        )
        dot_color = LOCAL_COLORS["train"] if is_train else LOCAL_COLORS["bus"]
        ax.scatter(lon, lat, c=dot_color, s=60, zorder=7,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(label, xy=(lon, lat), xytext=(4, 4),
                    textcoords="offset points", fontsize=5.5,
                    color="white", alpha=0.85, zorder=8)

    # ── Start / End markers ──
    ax.scatter(start_coords[1], start_coords[0],
               c="#00e5c0", s=400, zorder=10, marker="*")
    ax.annotate("START", xy=(start_coords[1], start_coords[0]),
                xytext=(6, 6), textcoords="offset points",
                color="#00e5c0", fontsize=9, fontweight="bold", zorder=11)

    ax.scatter(end_coords[1], end_coords[0],
               c="#ff3333", s=400, zorder=10, marker="*")
    ax.annotate("END", xy=(end_coords[1], end_coords[0]),
                xytext=(6, 6), textcoords="offset points",
                color="#ff3333", fontsize=9, fontweight="bold", zorder=11)

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0a1628")
    plt.close(fig)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Hero header ──
    st.markdown("""
    <div class="hero">
      <h1>🚇 Mumbai Multimodal Planner</h1>
      <p>Find the fastest route using any combination of Mumbai's transport network</p>
      <div style="margin-top:10px;">
        <span class="mode-badge">⚡ Earliest Arrival</span>
        <span class="mode-badge">🔄 Least Interchange</span>
        <span class="mode-badge">🚆 Local Train</span>
        <span class="mode-badge">🚇 Metro &amp; Monorail</span>
        <span class="mode-badge">🚌 BEST Bus</span>
        <span class="mode-badge">🚖 Cab</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    try:
        data   = load_data()
        router = get_router(data)
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}")
        st.stop()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.subheader("📍 Start Location")
        start_lat = st.number_input("Latitude",  value=19.0760, format="%.6f", step=0.0001, key="slat")
        start_lon = st.number_input("Longitude", value=72.8777, format="%.6f", step=0.0001, key="slon")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.subheader("🔴 Destination")
        end_lat = st.number_input("Latitude",  value=19.1972, format="%.6f", step=0.0001, key="elat")
        end_lon = st.number_input("Longitude", value=72.9780, format="%.6f", step=0.0001, key="elon")
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("🚦 Travel Mode")
        mode = st.selectbox(
            "Mode",
            options=["earliest_arrival", "least_interchange", "train", "metro", "bus", "car"],
            format_func=lambda m: {
                "earliest_arrival":  "⚡ Earliest Arrival (all modes)",
                "least_interchange": "🔄 Least Interchange (all modes)",
                "train":             "🚆 Local Train only",
                "metro":             "🚇 Metro & Monorail only",
                "bus":               "🚌 Bus only",
                "car":               "🚖 Cab only",
            }[m],
        )

        compute = st.button("🔍 Find Best Route", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        **Speed model**
        <div style='font-size:0.82rem;color:#9aa7c4;margin-top:4px;'>
        🚶 Walk: 12 min/km<br>
        🚌 Bus: 4 min/km<br>
        🚆 Train: timetable data<br>
        🚖 Cab: 3 min/km (min 1 km)
        </div>
        """, unsafe_allow_html=True)

    # ── Landing / pre-compute ──
    if not compute:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class='feature-card'>
              <div class='feature-icon'>⚡</div>
              <div class='feature-title'>Earliest Arrival</div>
              <p class='feature-desc'>Dijkstra across all modes — local train, metro,
              monorail, bus, cab and walk — minimising total journey time.</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='feature-card'>
              <div class='feature-icon'>🔄</div>
              <div class='feature-title'>Least Interchange</div>
              <p class='feature-desc'>Uses all modes but penalises every line or
              mode change heavily, so you transfer as few times as possible.</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='feature-card'>
              <div class='feature-icon'>🚇</div>
              <div class='feature-title'>Metro &amp; Monorail</div>
              <p class='feature-desc'>Isolates the metro network — Lines 1, 2A, 3, 7,
              Navi Mumbai Metro and Mumbai Monorail — for quick urban hops.</p>
            </div>""", unsafe_allow_html=True)
        st.info("👈 Set your coordinates in the sidebar and click **Find Best Route**.")
        return

    # ── Route computation ──
    start_coords = (start_lat, start_lon)
    end_coords   = (end_lat,   end_lon)

    mode_labels = {
        "earliest_arrival":  "Earliest Arrival",
        "least_interchange": "Least Interchange",
        "train": "Local Train", "metro": "Metro & Monorail",
        "bus": "Bus", "car": "Cab",
    }
    with st.spinner(f"Computing {mode_labels.get(mode, mode)} route…"):
        try:
            result = router.route(start_coords, end_coords, mode)
        except Exception as e:
            st.error(f"Routing error: {e}")
            st.exception(e)
            return

    if result[0] is None:
        st.error("No route found between the selected locations.")
        if result[3]:
            for a in result[3]:
                st.warning(a)
        return

    path, steps, total_time, advisories, G_multi = result

    # ── Summary ──
    total_dist = sum(s["distance_km"] for s in steps)
    mode_times = {}
    for s in steps:
        mode_times[s["mode"]] = mode_times.get(s["mode"], 0) + s["time_min"]

    st.markdown(f"""
    <div class="success-banner">
      <div>
        <div class="big">✅ Route Found!</div>
        <div class="small">Shortest-time path across Mumbai's network</div>
      </div>
      <div>
        <div class="big">⏱ {total_time:.1f} min</div>
        <div class="small">total journey time</div>
      </div>
      <div>
        <div class="big">📏 {total_dist:.2f} km</div>
        <div class="small">total distance</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode breakdown metrics ──
    if len(mode_times) > 1:
        st.markdown("**Time breakdown by mode**")
        bcols = st.columns(len(mode_times))
        for i, (m, t) in enumerate(mode_times.items()):
            bcols[i].metric(
                f"{MODE_EMOJI.get(m,'')} {m.title()}",
                f"{t:.1f} min",
                f"{t/total_time*100:.0f}%",
            )

    # ── Progress bar ──
    all_legs  = group_into_legs(steps)
    disp_legs = legs_for_display(all_legs)
    render_progress_bar(all_legs, total_time)

    # Legend for progress bar
    legend_mode_html = []
    for m in dict.fromkeys(l["mode"] for l in all_legs):
        color = MODE_HEX.get(m, "#888")
        legend_mode_html.append(
            f"<span class='legend-item'>"
            f"<span class='legend-dot' style='background:{color};'></span>"
            f"{m.title()}</span>"
        )
    st.markdown(" ".join(legend_mode_html), unsafe_allow_html=True)

    # ── Advisories ──
    if advisories:
        unique_adv = list(dict.fromkeys(advisories))
        with st.expander(f"⚠️ Advisories ({len(unique_adv)})", expanded=False):
            for a in unique_adv:
                st.warning(a)

    st.divider()

    # ── Two-column layout ──
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("📋 Route Steps")
        for i, leg in enumerate(disp_legs):
            render_leg_card(leg, i, data)

    with right:
        st.subheader("🗺️ Route Map")
        with st.spinner("Rendering map…"):
            try:
                buf = render_map(steps, start_coords, end_coords, data.G_road, data)
                st.image(buf, use_container_width=True)
                # HTML legend below map
                legend_items = "".join(
                    f"<div class='legend-item'>"
                    f"<div class='legend-dot' style='background:{c};'></div>"
                    f"<span>{m.title()}</span></div>"
                    for m, c in {
                        "Walk":  "#aaaaaa",
                        "Train": "#ff3333",
                        "Bus":   "#1e90ff",
                        "Cab":   "#ffd700",
                    }.items()
                )
                legend_items += (
                    "<div class='legend-item'>"
                    "<div class='legend-dot' style='background:#00e5c0;border-radius:50%;'></div>"
                    "<span>Start</span></div>"
                    "<div class='legend-item'>"
                    "<div class='legend-dot' style='background:#ff3333;border-radius:50%;'></div>"
                    "<span>End</span></div>"
                )
                st.markdown(f"<div class='map-legend'>{legend_items}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Map rendering failed: {e}")


if __name__ == "__main__":
    main()