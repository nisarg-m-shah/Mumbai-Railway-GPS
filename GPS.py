"""
Mumbai Multimodal Route Planner
================================
Modes: Train | Bus | Car | Combination
Algorithm: Weighted shortest-path on unified multigraph (NetworkX)

Required files:
  - mumbai_graph.graphml         (OSMnx road graph)
  - Cleaned Local Train Dataset.csv
  - railway_mumbai.csv
  - Bus Dataset 3.csv

Install deps:
  pip install osmnx networkx pandas matplotlib folium scipy
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import re
import time
import math
import warnings
from difflib import get_close_matches
from functools import lru_cache

import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

warnings.filterwarnings("ignore")

# Directory containing this script — used to resolve data files regardless of
# the working directory the process was launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))

def _data_path(filename: str) -> str:
    return os.path.join(_HERE, filename)


# ──────────────────────────────────────────────────────────────────────────────
# TIME MODEL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
WALK_MIN_PER_KM   = 12.0   # minutes per km for walking
BUS_MIN_PER_KM    = 4.0    # minutes per km for bus travel
CAR_MIN_PER_KM    = 3.0    # minutes per km for cab
CAB_MIN_DIST_KM   = 1.0    # cab not allowed below this distance
MAX_TRANSFER_DIST_KM = 0.5 # max walking distance for bus↔train transfer
TRAIN_LINE_PENALTY = 5.0   # minutes for same-station line change
BUS_ROUTE_LIMIT   = 10     # max nearby bus routes to include

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, strip bracketed text, remove junction/jn, collapse spaces."""
    name = str(name).lower().strip()
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace("junction", "").replace("jn", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def road_distance_km(G_road, node_a, node_b) -> float:
    """Shortest road distance in km between two OSMnx node IDs."""
    try:
        length_m = nx.shortest_path_length(G_road, node_a, node_b, weight="length")
        return length_m / 1000.0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def nearest_road_node(G_road, lat, lon):
    """Return the OSMnx node closest to (lat, lon). Cached by rounded coords."""
    return ox.distance.nearest_nodes(G_road, lon, lat)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

class MumbaiData:
    """Loads and exposes all static datasets."""

    def __init__(self):
        print("Loading road graph …")
        self.G_road = ox.load_graphml(_data_path("mumbai_graph.graphml"))

        print("Loading train data …")
        self._load_trains()

        print("Loading bus data …")
        self._load_buses()

        print("All data loaded.\n")

    # ── Train ──────────────────────────────────────────────────────────────

    def _load_trains(self):
        df = pd.read_csv(_data_path("Cleaned Local Train Dataset.csv"))
        df["station_clean"] = df["Station"].apply(normalize)

        # Build rail graph with time weight
        G_rail = nx.Graph()
        for _, row in df.iterrows():
            G_rail.add_node(row["station_clean"], original=row["Station"])

        for line, group in df.groupby("Line", sort=False):
            group = group.reset_index(drop=True)
            for i in range(1, len(group)):
                prev = group.loc[i - 1, "station_clean"]
                curr = group.loc[i,     "station_clean"]
                t    = group.loc[i,     "Time taken From Previous of the Line"]
                d    = group.loc[i,     "Distance From Previous of the Line"]
                G_rail.add_edge(prev, curr,
                                weight=float(t),
                                distance_km=float(d),
                                line=line,
                                mode="train")

        self.G_rail   = G_rail
        self.train_df = df

        # ── Fuzzy-match POI coordinates → rail graph station names ───────
        # The two datasets use slightly different spellings / extra words
        # (e.g. "Dadar" vs "Dadar Junction"), so we:
        #   1. Try exact match on normalized name  (fast path)
        #   2. Fall back to fuzzy difflib match    (handles typos, dropped words)
        pois = pd.read_csv(_data_path("railway_mumbai.csv"))
        pois["poi_clean"] = pois["name"].apply(normalize)

        rail_names = list(G_rail.nodes)
        poi_names  = pois["poi_clean"].tolist()

        station_coords  = {}
        station_display = {}
        fuzzy_log       = []

        for sc in rail_names:
            # 1. exact
            match = pois[pois["poi_clean"] == sc]
            if not match.empty:
                row = match.iloc[0]
                station_coords[sc]  = (float(row["latitude"]), float(row["longitude"]))
                station_display[sc] = str(row["name"])
                continue

            # 2. fuzzy  (cutoff 0.72 tolerates one-word differences and minor typos)
            hits = get_close_matches(sc, poi_names, n=1, cutoff=0.72)
            if hits:
                row = pois[pois["poi_clean"] == hits[0]].iloc[0]
                station_coords[sc]  = (float(row["latitude"]), float(row["longitude"]))
                station_display[sc] = str(row["name"])
                fuzzy_log.append(f"  fuzzy: {sc!r:35s} -> {hits[0]!r}")

        if fuzzy_log:
            print(f"Fuzzy-matched {len(fuzzy_log)} train station name(s):")
            for msg in fuzzy_log:
                print(msg)

        matched = len(station_coords)
        total   = len(rail_names)
        print(f"Train coords resolved: {matched}/{total} stations "
              f"({total - matched} unmatched, will be skipped in routing)")

        self.station_coords      = station_coords
        self.station_display     = station_display
        self.station_names_clean = list(station_coords.keys())

    # ── Bus ───────────────────────────────────────────────────────────────

    def _load_buses(self):
        df = pd.read_csv(_data_path("Bus Dataset 3.csv"))
        df = df.dropna(subset=["stop_lat", "stop_lon", "stop_id", "route_short_name"])
        df["stop_id"] = df["stop_id"].astype(str)
        self.bus_df = df

        # Build stop coordinate lookup
        stops = df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].drop_duplicates("stop_id")
        self.stop_coords = {
            row["stop_id"]: (float(row["stop_lat"]), float(row["stop_lon"]))
            for _, row in stops.iterrows()
        }


# ──────────────────────────────────────────────────────────────────────────────
# MULTIMODAL GRAPH BUILDER
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalGraphBuilder:
    """
    Builds the unified graph G_multi used for shortest-path routing.
    Node naming:
        train_<station_clean>
        bus_<stop_id>
        start / end
    """

    def __init__(self, data: MumbaiData, mode: str,
                 start_coords: tuple, end_coords: tuple):
        self.data  = data
        self.mode  = mode  # "train" | "bus" | "car" | "combination"
        self.start_latlon = start_coords  # (lat, lon)
        self.end_latlon   = end_coords

        # MultiDiGraph allows parallel edges (e.g. both walk AND cab between same
        # node pair, letting Dijkstra pick the cheaper one)
        self.G_multi = nx.MultiDiGraph()
        self._road_node_cache = {}
        self.advisories = []

    # ── Road node helpers ─────────────────────────────────────────────────

    def _road_node(self, lat, lon):
        key = (round(lat, 5), round(lon, 5))
        if key not in self._road_node_cache:
            self._road_node_cache[key] = nearest_road_node(self.data.G_road, lat, lon)
        return self._road_node_cache[key]

    def _road_dist_km(self, lat1, lon1, lat2, lon2) -> float:
        """Road distance in km via the OSMnx graph (cached node lookup)."""
        n1 = self._road_node(lat1, lon1)
        n2 = self._road_node(lat2, lon2)
        return road_distance_km(self.data.G_road, n1, n2)

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self) -> nx.DiGraph:
        G = self.G_multi
        mode = self.mode

        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        G.add_node("start", lat=s_lat, lon=s_lon)
        G.add_node("end",   lat=e_lat, lon=e_lon)

        if mode == "car":
            self._build_car_only()
        elif mode == "train":
            self._build_train_layer()
            self._connect_endpoints_to_train()
        elif mode == "bus":
            self._build_bus_layer()
            self._connect_endpoints_to_bus()
        elif mode == "combination":
            self._build_train_layer()
            self._build_bus_layer()
            self._build_transfers()
            self._connect_endpoints_combination()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return G

    # ── Road-distance helper (uses the actual road graph) ─────────────────

    def _road_dist_and_time(self, lat1, lon1, lat2, lon2, speed_min_per_km):
        """
        Compute road distance (km) and travel time (min) between two lat/lon
        points by looking up the road graph.  Returns (dist_km, time_min).
        The road graph IS the physical network for both walking and cab — only
        the speed constant differs.  No synthetic edges are added to G_multi.
        """
        d = self._road_dist_km(lat1, lon1, lat2, lon2)
        return d, d * speed_min_per_km

    # ── Car-only ──────────────────────────────────────────────────────────

    def _build_car_only(self):
        """
        Car mode: single edge start→end whose weight = road_distance × CAR_MIN_PER_KM.
        The road graph provides the distance; no separate 'car graph' is needed.
        """
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon
        d_km, t = self._road_dist_and_time(s_lat, s_lon, e_lat, e_lon, CAR_MIN_PER_KM)
        self.G_multi.add_edge("start", "end",
                              weight=t, distance_km=d_km,
                              mode="car",
                              seg_start=(s_lat, s_lon),
                              seg_end=(e_lat, e_lon))

    # ── Train layer ───────────────────────────────────────────────────────

    def _build_train_layer(self):
        """Add all rail edges (bidirectional) with time weight from CSV."""
        G    = self.G_multi
        data = self.data

        for u, v, attr in data.G_rail.edges(data=True):
            nu = f"train_{u}"
            nv = f"train_{v}"
            u_coords = data.station_coords.get(u)
            v_coords = data.station_coords.get(v)
            G.add_node(nu, lat=u_coords[0] if u_coords else None,
                           lon=u_coords[1] if u_coords else None,
                           station=u)
            G.add_node(nv, lat=v_coords[0] if v_coords else None,
                           lon=v_coords[1] if v_coords else None,
                           station=v)
            for src, dst, sc, dc in [(nu, nv, u_coords, v_coords),
                                      (nv, nu, v_coords, u_coords)]:
                G.add_edge(src, dst,
                           weight=attr["weight"], mode="train",
                           distance_km=attr.get("distance_km", 0),
                           line=attr.get("line", ""),
                           seg_start=sc, seg_end=dc)

    def _connect_endpoints_to_train(self):
        """
        Connect start/end → the N_TRAIN_CANDIDATES nearest stations.

        Two edges are added per (endpoint, station) pair:
          1. Walk edge  — weight = road_dist × WALK_MIN_PER_KM  (always)
          2. Cab edge   — weight = road_dist × CAR_MIN_PER_KM   (only if road_dist ≥ 1 km)

        Both edges traverse the same physical road graph; only the speed differs.
        The shortest-path algorithm then picks whichever is cheaper in total time.
        An advisory is emitted when the walk distance exceeds 1 km, as per spec §4.
        """
        N_TRAIN_CANDIDATES = 5   # nearest stations per endpoint; enough for good coverage
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        for ep, lat, lon in [("start", s_lat, s_lon), ("end", e_lat, e_lon)]:
            # Haversine pre-sort → cheap O(N) filter before expensive road-dist calls
            station_items = sorted(
                self.data.station_coords.items(),
                key=lambda x: haversine_km(lat, lon, x[1][0], x[1][1])
            )
            candidates = station_items[:N_TRAIN_CANDIDATES]

            for sc, (st_lat, st_lon) in candidates:
                node = f"train_{sc}"
                if node not in self.G_multi:
                    continue

                d_km, walk_t = self._road_dist_and_time(lat, lon, st_lat, st_lon, WALK_MIN_PER_KM)
                if d_km == float("inf"):
                    continue

                _, cab_t = d_km, d_km * CAR_MIN_PER_KM

                if ep == "start":
                    self.G_multi.add_edge("start", node,
                                         weight=walk_t, mode="walk",
                                         distance_km=d_km,
                                         seg_start=(lat, lon),
                                         seg_end=(st_lat, st_lon))
                    if d_km >= CAB_MIN_DIST_KM:
                        self.G_multi.add_edge("start", node,
                                             weight=cab_t, mode="car",
                                             distance_km=d_km,
                                             seg_start=(lat, lon),
                                             seg_end=(st_lat, st_lon))
                        self.advisories.append(
                            f"Walk to {sc} is {d_km:.2f} km — cab recommended for this leg."
                        )
                else:
                    self.G_multi.add_edge(node, "end",
                                         weight=walk_t, mode="walk",
                                         distance_km=d_km,
                                         seg_start=(st_lat, st_lon),
                                         seg_end=(lat, lon))
                    if d_km >= CAB_MIN_DIST_KM:
                        self.G_multi.add_edge(node, "end",
                                             weight=cab_t, mode="car",
                                             distance_km=d_km,
                                             seg_start=(st_lat, st_lon),
                                             seg_end=(lat, lon))
                        self.advisories.append(
                            f"Walk from {sc} to destination is {d_km:.2f} km — cab recommended."
                        )

    # ── Bus layer ─────────────────────────────────────────────────────────

    def _get_relevant_routes(self) -> pd.DataFrame:
        """
        Per spec §7: collect the 10 nearest distinct routes near START and the
        10 nearest distinct routes near END (by haversine to any stop on the route),
        then take the union.  All stops on those routes are included so Dijkstra
        can traverse each route end-to-end without artificial gaps.

        This is still greedy/limited (intentionally, for performance), but the
        union of start-side + end-side routes means a route that serves both areas
        is not missed.
        """
        bus_df      = self.data.bus_df
        stops_uniq  = (bus_df[["stop_id","stop_lat","stop_lon","route_short_name"]]
                       .drop_duplicates("stop_id"))

        def nearest_routes(lat, lon, limit):
            dists = stops_uniq.apply(
                lambda r: haversine_km(lat, lon, float(r["stop_lat"]), float(r["stop_lon"])),
                axis=1
            )
            sorted_stops = stops_uniq.assign(dist=dists).sort_values("dist")
            seen, routes = set(), []
            for _, row in sorted_stops.iterrows():
                r = row["route_short_name"]
                if r not in seen:
                    seen.add(r)
                    routes.append(r)
                if len(routes) >= limit:
                    break
            return set(routes)

        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon
        routes = (nearest_routes(s_lat, s_lon, BUS_ROUTE_LIMIT) |
                  nearest_routes(e_lat, e_lon, BUS_ROUTE_LIMIT))
        return bus_df[bus_df["route_short_name"].isin(routes)]

    def _build_bus_layer(self):
        """Add sequential bus edges (haversine distance × BUS_MIN_PER_KM) for filtered routes."""
        G          = self.G_multi
        bus_subset = self._get_relevant_routes()

        for route, grp in bus_subset.groupby("route_short_name"):
            grp = grp.sort_values("stop_sequence").reset_index(drop=True)
            for i in range(1, len(grp)):
                prev, curr = grp.loc[i-1], grp.loc[i]
                n_prev = f"bus_{prev['stop_id']}"
                n_curr = f"bus_{curr['stop_id']}"
                G.add_node(n_prev, lat=float(prev["stop_lat"]), lon=float(prev["stop_lon"]),
                           stop_name=prev["stop_name"])
                G.add_node(n_curr, lat=float(curr["stop_lat"]), lon=float(curr["stop_lon"]),
                           stop_name=curr["stop_name"])
                d = haversine_km(float(prev["stop_lat"]), float(prev["stop_lon"]),
                                 float(curr["stop_lat"]), float(curr["stop_lon"]))
                G.add_edge(n_prev, n_curr,
                           weight=d * BUS_MIN_PER_KM, mode="bus",
                           distance_km=d, route=route,
                           seg_start=(float(prev["stop_lat"]), float(prev["stop_lon"])),
                           seg_end=(float(curr["stop_lat"]),   float(curr["stop_lon"])))

        self._bus_subset = bus_subset

    def _connect_endpoints_to_bus(self):
        """
        Connect start/end → the N_BUS_CANDIDATES nearest bus stops (from the
        filtered set already in G_multi).

        Same dual-edge logic as train: walk edge always, cab edge when ≥ 1 km.
        The cab edge weight is road_dist × CAR_MIN_PER_KM; both edges use the
        road graph for distance — no synthetic road network is created.
        """
        N_BUS_CANDIDATES = 5
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        bus_nodes = [(n, a) for n, a in self.G_multi.nodes(data=True)
                     if n.startswith("bus_") and a.get("lat") is not None]

        for ep, lat, lon in [("start", s_lat, s_lon), ("end", e_lat, e_lon)]:
            candidates = sorted(
                bus_nodes,
                key=lambda x: haversine_km(lat, lon, x[1]["lat"], x[1]["lon"])
            )[:N_BUS_CANDIDATES]

            for bn, ba in candidates:
                d_km, walk_t = self._road_dist_and_time(lat, lon, ba["lat"], ba["lon"],
                                                        WALK_MIN_PER_KM)
                if d_km == float("inf"):
                    continue

                cab_t = d_km * CAR_MIN_PER_KM

                if ep == "start":
                    self.G_multi.add_edge("start", bn,
                                         weight=walk_t, mode="walk",
                                         distance_km=d_km,
                                         seg_start=(lat, lon),
                                         seg_end=(ba["lat"], ba["lon"]))
                    if d_km >= CAB_MIN_DIST_KM:
                        self.G_multi.add_edge("start", bn,
                                             weight=cab_t, mode="car",
                                             distance_km=d_km,
                                             seg_start=(lat, lon),
                                             seg_end=(ba["lat"], ba["lon"]))
                        self.advisories.append(
                            f"Walk to bus stop is {d_km:.2f} km — cab recommended for this leg."
                        )
                else:
                    self.G_multi.add_edge(bn, "end",
                                         weight=walk_t, mode="walk",
                                         distance_km=d_km,
                                         seg_start=(ba["lat"], ba["lon"]),
                                         seg_end=(lat, lon))
                    if d_km >= CAB_MIN_DIST_KM:
                        self.G_multi.add_edge(bn, "end",
                                             weight=cab_t, mode="car",
                                             distance_km=d_km,
                                             seg_start=(ba["lat"], ba["lon"]),
                                             seg_end=(lat, lon))
                        self.advisories.append(
                            f"Walk from bus stop to destination is {d_km:.2f} km — cab recommended."
                        )

    # ── Transfers (combination mode) ──────────────────────────────────────

    def _build_transfers(self):
        """
        Wire up inter-layer transfer edges for combination mode.

        Bus ↔ Train  : walking transfer (road dist × WALK_MIN_PER_KM)
                       only when haversine distance ≤ MAX_TRANSFER_DIST_KM (0.5 km)
        Train ↔ Train: 5-minute penalty for same station, different line
        Cab (start/end → any transit node): road_dist × CAR_MIN_PER_KM, if ≥ 1 km.
                       The cab edge is a virtual edge in G_multi whose weight
                       encodes the time cost of a road-graph traversal at car speed.
        """
        G = self.G_multi

        train_nodes = [(n, a) for n, a in G.nodes(data=True)
                       if n.startswith("train_") and a.get("lat")]
        bus_nodes   = [(n, a) for n, a in G.nodes(data=True)
                       if n.startswith("bus_")   and a.get("lat")]

        # ── Bus ↔ Train walk transfers ──
        for tn, ta in train_nodes:
            for bn, ba in bus_nodes:
                if haversine_km(ta["lat"], ta["lon"], ba["lat"], ba["lon"]) <= MAX_TRANSFER_DIST_KM:
                    d_km, t = self._road_dist_and_time(
                        ta["lat"], ta["lon"], ba["lat"], ba["lon"], WALK_MIN_PER_KM)
                    for src, dst, sc, dc in [(tn, bn, (ta["lat"],ta["lon"]), (ba["lat"],ba["lon"])),
                                             (bn, tn, (ba["lat"],ba["lon"]), (ta["lat"],ta["lon"]))]:
                        G.add_edge(src, dst, weight=t, mode="walk",
                                   distance_km=d_km, seg_start=sc, seg_end=dc)

        # ── Train ↔ Train same-station line penalty ──
        by_station = {}
        for tn, ta in train_nodes:
            by_station.setdefault(ta.get("station",""), []).append((tn, ta))
        for sc, node_attrs in by_station.items():
            for i in range(len(node_attrs)):
                for j in range(i+1, len(node_attrs)):
                    ni, ai = node_attrs[i]
                    nj, aj = node_attrs[j]
                    ci = (ai["lat"], ai["lon"]) if ai.get("lat") else None
                    cj = (aj["lat"], aj["lon"]) if aj.get("lat") else None
                    # These ARE walk legs (changing platform) with real coordinates
                    G.add_edge(ni, nj, weight=TRAIN_LINE_PENALTY,
                               mode="walk", distance_km=0, seg_start=ci, seg_end=cj)
                    G.add_edge(nj, ni, weight=TRAIN_LINE_PENALTY,
                               mode="walk", distance_km=0, seg_start=cj, seg_end=ci)

        # ── Cab option: start/end ↔ nearest transit nodes (combination only) ──
        # In combination mode the user CAN take a cab for any leg. We add cab-speed
        # edges from start→transit and transit→end for the same N candidates used
        # by the walk connections above, so Dijkstra can pick cab mid-route too.
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        # Direct start→end cab (if ≥ 1 km) — lets Dijkstra pick pure cab when fastest
        d_km, cab_t = self._road_dist_and_time(s_lat, s_lon, e_lat, e_lon, CAR_MIN_PER_KM)
        if d_km >= CAB_MIN_DIST_KM:
            G.add_edge("start", "end", weight=cab_t, mode="car",
                       distance_km=d_km,
                       seg_start=(s_lat, s_lon), seg_end=(e_lat, e_lon))

    def _connect_endpoints_combination(self):
        """Walk + cab connections for both layers, then inter-layer transfers."""
        self._connect_endpoints_to_train()
        self._connect_endpoints_to_bus()


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────

class Router:

    def __init__(self, data: MumbaiData):
        self.data = data

    def route(self, start_coords, end_coords, mode):
        builder = MultimodalGraphBuilder(self.data, mode, start_coords, end_coords)
        G_multi = builder.build()

        if not nx.has_path(G_multi, "start", "end"):
            return None, None, None, builder.advisories

        path = nx.shortest_path(G_multi, "start", "end", weight="weight")
        total_time = nx.shortest_path_length(G_multi, "start", "end", weight="weight")

        steps = self._extract_steps(G_multi, path)
        return path, steps, total_time, builder.advisories, G_multi

    def _extract_steps(self, G, path):
        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # MultiDiGraph: G[u][v] is a dict of {edge_key: attr_dict}
            # Pick the edge with the minimum weight (walk vs cab parallel edges)
            edge_data = G[u][v]
            attr = min(edge_data.values(), key=lambda a: a.get("weight", float("inf")))
            steps.append({
                "from":        u,
                "to":          v,
                "mode":        attr.get("mode", "?"),
                "distance_km": attr.get("distance_km", 0),
                "time_min":    attr.get("weight", 0),
                "route":       attr.get("route", attr.get("line", "")),
                "seg_start":   attr.get("seg_start"),
                "seg_end":     attr.get("seg_end"),
            })
        return steps


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZER
# ──────────────────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "walk": "gray",
    "train": "red",
    "bus": "blue",
    "car": "green",
}

class Visualizer:

    def __init__(self, data: MumbaiData):
        self.data = data

    def plot(self, steps, start_coords, end_coords, title="Mumbai Multimodal Route"):
        G_road = self.data.G_road
        fig, ax = ox.plot_graph(G_road, show=False, close=False,
                                node_size=0,
                                edge_color="#cccccc",
                                edge_linewidth=0.3,
                                bgcolor="white")

        for step in steps:
            color = MODE_COLORS.get(step["mode"], "purple")
            seg_s = step.get("seg_start")
            seg_e = step.get("seg_end")

            if seg_s is None or seg_e is None:
                continue

            if step["mode"] == "train":
                # Straight line between station coords
                ax.plot([seg_s[1], seg_e[1]], [seg_s[0], seg_e[0]],
                        color=color, linewidth=3, alpha=0.85, zorder=4)
            else:
                # Road geometry
                try:
                    n1 = ox.distance.nearest_nodes(G_road, seg_s[1], seg_s[0])
                    n2 = ox.distance.nearest_nodes(G_road, seg_e[1], seg_e[0])
                    road_path = nx.shortest_path(G_road, n1, n2, weight="length")
                    lons = [G_road.nodes[n]["x"] for n in road_path]
                    lats = [G_road.nodes[n]["y"] for n in road_path]
                    ax.plot(lons, lats, color=color, linewidth=3, alpha=0.85, zorder=4)
                except Exception:
                    ax.plot([seg_s[1], seg_e[1]], [seg_s[0], seg_e[0]],
                            color=color, linewidth=2, linestyle="--", alpha=0.7, zorder=4)

        # Start / End markers
        ax.scatter(start_coords[1], start_coords[0], c="lime",   s=150, zorder=6, label="Start")
        ax.scatter(end_coords[1],   end_coords[0],   c="orange", s=150, zorder=6, label="End")

        # Legend
        patches = [mpatches.Patch(color=c, label=m.title()) for m, c in MODE_COLORS.items()]
        patches += [mpatches.Patch(color="lime",   label="Start"),
                    mpatches.Patch(color="orange", label="End")]
        ax.legend(handles=patches, loc="upper left", fontsize=8)

        plt.title(title)
        plt.tight_layout()
        plt.savefig("route_map.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Map saved to route_map.png")


# ──────────────────────────────────────────────────────────────────────────────
# KNOWN PLACES (predefined list with lat/lon)
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_PLACES = {
    "Chhatrapati Shivaji Maharaj Terminus (CSMT)": (18.9398, 72.8354),
    "Dadar": (19.0178, 72.8478),
    "Andheri": (19.1196, 72.8470),
    "Bandra": (19.0596, 72.8295),
    "Kurla": (19.0728, 72.8826),
    "Thane": (19.1972, 72.9780),
    "Borivali": (19.2307, 72.8567),
    "Churchgate": (18.9322, 72.8264),
    "Chembur": (19.0622, 72.9005),
    "Powai": (19.1176, 72.9060),
    "Juhu Beach": (19.1075, 72.8263),
    "BKC (Bandra Kurla Complex)": (19.0660, 72.8654),
    "Nariman Point": (18.9256, 72.8242),
    "Colaba": (18.9067, 72.8147),
    "Ghatkopar": (19.0863, 72.9081),
    "Vikhroli": (19.1059, 72.9300),
    "Malad": (19.1871, 72.8487),
    "Goregaon": (19.1663, 72.8526),
    "Vile Parle": (19.0990, 72.8490),
    "Santacruz": (19.0828, 72.8408),
    "Chembur Station": (19.0580, 72.8990),
    "Worli": (19.0176, 72.8155),
    "Lower Parel": (18.9952, 72.8261),
    "Mulund": (19.1726, 72.9560),
    "Belapur (CBD)": (19.0230, 73.0298),
    "Vashi": (19.0771, 73.0071),
    "Panvel": (18.9894, 73.1175),
    "Airport (T2)": (19.0990, 72.8677),
    "Elphinstone Road": (18.9962, 72.8176),
    "Matunga": (19.0270, 72.8593),
}


def select_place(prompt_text):
    names = list(KNOWN_PLACES.keys())
    print(f"\n{prompt_text}")
    for i, name in enumerate(names):
        print(f"  {i:2d}. {name}")
    while True:
        try:
            idx = int(input("Enter number: "))
            if 0 <= idx < len(names):
                selected = names[idx]
                return selected, KNOWN_PLACES[selected]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def select_mode():
    modes = ["train", "bus", "car", "combination"]
    print("\nSelect travel mode:")
    for i, m in enumerate(modes):
        print(f"  {i}. {m}")
    while True:
        try:
            idx = int(input("Enter number: "))
            if 0 <= idx < len(modes):
                return modes[idx]
        except ValueError:
            pass
        print("Invalid. Try again.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def print_route(steps, total_time, advisories):
    print("\n" + "="*60)
    print("  ROUTE SUMMARY")
    print("="*60)

    mode_time = {}
    for step in steps:
        m = step["mode"]
        mode_time[m] = mode_time.get(m, 0) + step["time_min"]
        route_info = f"  [{step['route']}]" if step.get("route") else ""
        print(f"  {step['mode'].upper():6s}{route_info:20s}  "
              f"{step['distance_km']:5.2f} km  "
              f"{step['time_min']:5.1f} min  "
              f"{step['from']} → {step['to']}")

    print("-"*60)
    print(f"  TOTAL TIME: {total_time:.1f} minutes")
    print("\n  Breakdown by mode:")
    for m, t in mode_time.items():
        print(f"    {m.upper():8s}: {t:.1f} min")

    if advisories:
        print("\n  ⚠️  Advisories:")
        for a in advisories:
            print(f"    • {a}")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("  MUMBAI MULTIMODAL ROUTE PLANNER")
    print("="*60)

    data = MumbaiData()
    router = Router(data)
    viz = Visualizer(data)

    start_name, start_coords = select_place("Select START location:")
    end_name,   end_coords   = select_place("Select DESTINATION:")
    mode = select_mode()

    print(f"\nComputing {mode} route: {start_name} → {end_name} …")
    t0 = time.time()
    result = router.route(start_coords, end_coords, mode)
    elapsed = time.time() - t0

    if result[0] is None:
        print("\n❌ No route found between selected locations.")
        if result[3]:
            for a in result[3]:
                print(f"  • {a}")
        return

    path, steps, total_time, advisories, G_multi = result
    print(f"Route computed in {elapsed:.2f}s")

    print_route(steps, total_time, advisories)

    viz.plot(steps, start_coords, end_coords,
             title=f"{mode.title()} Route: {start_name} → {end_name}")


if __name__ == "__main__":
    main()