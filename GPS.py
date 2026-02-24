"""
Mumbai Multimodal Route Planner — Routing Engine
=================================================
Modes:
  train             — Mumbai Local Train only
  metro             — Mumbai Metro / Navi Mumbai Metro / Monorail only
  bus               — BEST Bus only
  car               — Cab only
  earliest_arrival  — All modes, minimise total travel time
  least_interchange — All modes, minimise line/mode changes
  public_transport  — All transit, walk-only access (no cab at all)
Algorithm: Weighted shortest-path on unified MultiDiGraph (NetworkX)

Required files (same directory):
  - mumbai_graph.graphml
  - Final Train Dataset.csv
  - Bus Dataset 3.csv
"""

import re
import time
import math
import warnings
import os

import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

def _data_path(filename: str) -> str:
    return os.path.join(_HERE, filename)


# ──────────────────────────────────────────────────────────────────────────────
# TIME MODEL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
WALK_MIN_PER_KM      = 12.0
BUS_MIN_PER_KM       = 4.0
CAR_MIN_PER_KM       = 3.0
CAB_MIN_DIST_KM      = 1.0
SNAP_RADIUS_KM       = 0.20  # within 200 m: zero-cost snap edge (no road graph)
WALK_RADIUS_KM       = 1.0   # stations within this radius: walk edge (haversine)
CAB_RADIUS_KM        = 8.0   # stations within this radius: cab edge (haversine)
                              # Covers ~8 km — enough to reach a better line
                              # (e.g. Worli→Cotton Green 5 km, Worli→Byculla 4.5 km)
ROAD_FACTOR          = 1.3
MAX_TRANSFER_DIST_KM = 0.5
TRAIN_LINE_PENALTY   = 5.0
BUS_ROUTE_LIMIT      = 10
INTERCHANGE_PENALTY  = 20.0  # extra minutes added per mode/line change in least_interchange mode

# System groupings for filtered modes
SYSTEMS_LOCAL_TRAIN = {"Mumbai Local Train"}
SYSTEMS_METRO       = {"Mumbai Metro", "Navi Mumbai Metro", "Mumbai Monorail"}
SYSTEMS_ALL         = SYSTEMS_LOCAL_TRAIN | SYSTEMS_METRO
WALK_ONLY_MODES     = {"public_transport"}  # modes that forbid cab access edges

# Lines that are ONE-WAY (ascending sequence only — no reverse edges)
DIRECTIONAL_LINES = {"Yellow Line (Line 2A)", "Red Line (Line 7)"}

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace("junction", "").replace("jn", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def road_distance_km(G_road, node_a, node_b) -> float:
    try:
        return nx.shortest_path_length(G_road, node_a, node_b, weight="length") / 1000.0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def nearest_road_node(G_road, lat, lon):
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
        """
        Load Final Train Dataset.csv.

        Each row is one station on one line.  Node key = "Station Name__Line"
        so the same physical station on two lines becomes two separate graph
        nodes (joined later by a transfer penalty edge).

        Coordinates come directly from the CSV (Latitude / Longitude columns).
        """
        df = pd.read_csv(_data_path("Final Train Dataset.csv"))

        # Normalise column names — strip leading/trailing whitespace
        df.columns = df.columns.str.strip()

        # Required columns (case-tolerant)
        col_map = {}
        for col in df.columns:
            lc = col.lower()
            if "station" in lc and "name" in lc:
                col_map["Station Name"] = col
            elif lc == "latitude":
                col_map["Latitude"] = col
            elif lc == "longitude":
                col_map["Longitude"] = col
            elif lc == "line":
                col_map["Line"] = col
            elif "sequence" in lc:
                col_map["Sequence"] = col
            elif "distance" in lc and "previous" in lc:
                col_map["Distance From Previous"] = col
            elif "time" in lc and "previous" in lc:
                col_map["Time From Previous"] = col
            elif "system" in lc:
                col_map["System"] = col

        df = df.rename(columns={v: k for k, v in col_map.items()})

        # Drop rows with no coordinates
        df = df.dropna(subset=["Latitude", "Longitude"])
        df["Latitude"]  = df["Latitude"].astype(float)
        df["Longitude"] = df["Longitude"].astype(float)
        df["Sequence"]  = df["Sequence"].astype(int)
        df["Time From Previous"]     = pd.to_numeric(df["Time From Previous"],     errors="coerce").fillna(0)
        df["Distance From Previous"] = pd.to_numeric(df["Distance From Previous"], errors="coerce").fillna(0)

        self.train_df = df

        # station_coords  : "Station Name__Line" → (lat, lon)
        # station_display : "Station Name__Line" → readable label (station name only)
        station_coords  = {}
        station_display = {}

        for _, row in df.iterrows():
            key = f"{row['Station Name']}__{row['Line']}"
            station_coords[key]  = (row["Latitude"], row["Longitude"])
            station_display[key] = str(row["Station Name"])

        self.station_coords      = station_coords
        self.station_display     = station_display
        self.station_names_clean = list(station_coords.keys())

        total = len(station_coords)
        print(f"Train stations loaded: {total} (station, line) pairs from Final Train Dataset.csv")

    # ── Bus ───────────────────────────────────────────────────────────────

    def _load_buses(self):
        df = pd.read_csv(_data_path("Final Bus Dataset.csv"))
        df = df.dropna(subset=["stop_lat", "stop_lon", "stop_id", "route_short_name"])
        df["stop_id"] = df["stop_id"].astype(str)
        self.bus_df = df

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
    Node naming:
        train_<Station Name>__<Line>   — one node per (station, line) pair
        bus_<stop_id>
        start / end
    """

    def __init__(self, data: MumbaiData, mode: str,
                 start_coords: tuple, end_coords: tuple):
        self.data         = data
        self.mode         = mode
        self.start_latlon = start_coords
        self.end_latlon   = end_coords
        self.G_multi      = nx.MultiDiGraph()
        self._road_node_cache = {}
        self.advisories   = []

    # ── Road helpers ──────────────────────────────────────────────────────

    def _road_node(self, lat, lon):
        key = (round(lat, 5), round(lon, 5))
        if key not in self._road_node_cache:
            self._road_node_cache[key] = nearest_road_node(self.data.G_road, lat, lon)
        return self._road_node_cache[key]

    def _road_dist_km(self, lat1, lon1, lat2, lon2) -> float:
        n1 = self._road_node(lat1, lon1)
        n2 = self._road_node(lat2, lon2)
        return road_distance_km(self.data.G_road, n1, n2)

    def _road_dist_and_time(self, lat1, lon1, lat2, lon2, speed_min_per_km):
        d = self._road_dist_km(lat1, lon1, lat2, lon2)
        return d, d * speed_min_per_km

    # ── Cab distance helper (haversine × ROAD_FACTOR — no road graph call) ──

    def _cab_dist_and_time(self, lat1, lon1, lat2, lon2):
        h = haversine_km(lat1, lon1, lat2, lon2)
        d = h * ROAD_FACTOR
        return d, d * CAR_MIN_PER_KM

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self) -> nx.MultiDiGraph:
        G = self.G_multi
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        G.add_node("start", lat=s_lat, lon=s_lon)
        G.add_node("end",   lat=e_lat, lon=e_lon)

        if self.mode == "car":
            self._build_car_only()

        elif self.mode == "train":
            # Local trains only
            self._build_train_layer(allowed_systems=SYSTEMS_LOCAL_TRAIN)
            self._connect_endpoints_to_train()

        elif self.mode == "metro":
            # Metro + Monorail only (no local trains, no bus)
            self._build_train_layer(allowed_systems=SYSTEMS_METRO)
            self._connect_endpoints_to_train()

        elif self.mode == "bus":
            self._build_bus_layer()
            self._connect_endpoints_to_bus()

        elif self.mode in ("earliest_arrival", "least_interchange"):
            # Both use the full multimodal graph.
            # least_interchange adds INTERCHANGE_PENALTY to every transfer /
            # mode-change edge so Dijkstra prefers fewer interchanges.
            penalty = INTERCHANGE_PENALTY if self.mode == "least_interchange" else 0.0
            self._build_train_layer(allowed_systems=SYSTEMS_ALL,
                                    interchange_penalty=penalty)
            self._build_bus_layer()
            self._build_transfers(interchange_penalty=penalty)
            self._connect_endpoints_combination()

        elif self.mode == "public_transport":
            # All transit systems, walk-only first/last mile (no cab edges).
            # Minimises cab/walk distance by simply not offering cab as an option.
            self._build_train_layer(allowed_systems=SYSTEMS_ALL)
            self._build_bus_layer()
            self._build_transfers()
            self._connect_endpoints_walk_only()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return G

    # ── Car only ──────────────────────────────────────────────────────────

    def _build_car_only(self):
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon
        d_km, t = self._cab_dist_and_time(s_lat, s_lon, e_lat, e_lon)
        self.G_multi.add_edge("start", "end",
                              weight=t, distance_km=d_km, mode="car",
                              seg_start=(s_lat, s_lon), seg_end=(e_lat, e_lon))

    # ── Train layer ───────────────────────────────────────────────────────

    def _build_train_layer(self, allowed_systems=None, interchange_penalty=0.0):
        """
        Build rail edges.

        allowed_systems : set of System values to include (None = all).
                          e.g. SYSTEMS_LOCAL_TRAIN, SYSTEMS_METRO, SYSTEMS_ALL.
        interchange_penalty : extra minutes added to transfer/walk edges between
                              different lines at the same station (least_interchange mode).

        Forward edge  (seq i → seq i+1): weight = Time From Previous of i+1
        Reverse edge  (seq i+1 → seq i): same weight (symmetric travel time)
        Reverse edges SKIPPED for DIRECTIONAL_LINES (Yellow Line 2A, Red Line 7).
        """
        G   = self.G_multi
        df  = self.data.train_df

        # Filter by system if requested
        if allowed_systems is not None:
            df = df[df["System"].isin(allowed_systems)]

        for line, grp in df.groupby("Line"):
            grp = grp.sort_values("Sequence").reset_index(drop=True)
            is_directional = line in DIRECTIONAL_LINES

            for i in range(len(grp) - 1):
                row_i   = grp.iloc[i]
                row_i1  = grp.iloc[i + 1]

                # Skip stations with missing coordinates
                if pd.isna(row_i["Latitude"])  or pd.isna(row_i["Longitude"]):
                    continue
                if pd.isna(row_i1["Latitude"]) or pd.isna(row_i1["Longitude"]):
                    continue

                node_i  = f"train_{row_i['Station Name']}__{line}"
                node_i1 = f"train_{row_i1['Station Name']}__{line}"

                coords_i  = (float(row_i["Latitude"]),  float(row_i["Longitude"]))
                coords_i1 = (float(row_i1["Latitude"]), float(row_i1["Longitude"]))

                # Edge weight = Time From Previous of the destination
                t_fwd = float(row_i1["Time From Previous"])
                d_fwd = float(row_i1["Distance From Previous"])

                G.add_node(node_i,
                           lat=coords_i[0], lon=coords_i[1],
                           station=row_i["Station Name"], line=line,
                           system=row_i.get("System", ""))
                G.add_node(node_i1,
                           lat=coords_i1[0], lon=coords_i1[1],
                           station=row_i1["Station Name"], line=line,
                           system=row_i1.get("System", ""))

                # Resolve display mode from System column
                sys_val   = str(row_i.get("System", ""))
                edge_mode = SYSTEM_TO_MODE.get(sys_val, "train")

                # Forward edge (ascending)
                G.add_edge(node_i, node_i1,
                           weight=t_fwd, mode=edge_mode,
                           distance_km=d_fwd, line=line,
                           seg_start=coords_i, seg_end=coords_i1)

                if not is_directional:
                    # Reverse edge (descending): same time/distance
                    G.add_edge(node_i1, node_i,
                               weight=t_fwd, mode=edge_mode,
                               distance_km=d_fwd, line=line,
                               seg_start=coords_i1, seg_end=coords_i)

        # ── Transfer edges between lines at the same station ──────────────
        # Done here (not in _build_transfers) so single-mode rail builds also
        # have inter-line transfers (e.g. Western↔Central at Dadar).
        # In least_interchange mode, each transfer carries the extra penalty.
        train_nodes = [(n, a) for n, a in G.nodes(data=True)
                       if n.startswith("train_") and a.get("lat")]
        by_station = {}
        for tn, ta in train_nodes:
            by_station.setdefault(ta.get("station", ""), []).append((tn, ta))

        for station_name, node_attrs in by_station.items():
            for i in range(len(node_attrs)):
                for j in range(i + 1, len(node_attrs)):
                    ni, ai = node_attrs[i]
                    nj, aj = node_attrs[j]
                    # Skip if same line (no transfer needed)
                    if ai.get("line") == aj.get("line"):
                        continue
                    ci = (ai["lat"], ai["lon"]) if ai.get("lat") else None
                    cj = (aj["lat"], aj["lon"]) if aj.get("lat") else None
                    w  = TRAIN_LINE_PENALTY + interchange_penalty
                    G.add_edge(ni, nj, weight=w, mode="walk",
                               distance_km=0, seg_start=ci, seg_end=cj,
                               is_transfer=True)
                    G.add_edge(nj, ni, weight=w, mode="walk",
                               distance_km=0, seg_start=cj, seg_end=ci,
                               is_transfer=True)

    # ── Endpoint → train connections ──────────────────────────────────────

    def _connect_endpoints_to_train(self):
        """
        Connect start/end to rail stations using a search-radius model:

          Haversine <= WALK_RADIUS_KM (1 km) -> walk edge
          Haversine <= CAB_RADIUS_KM  (8 km) -> cab edge   (if >= CAB_MIN_DIST_KM)

        The 8 km cab radius is intentionally generous so that Dijkstra can
        consider strategically distant stations that save time overall —
        e.g. Worli -> Cotton Green (5 km cab) to board the Harbour Line
        directly rather than walking to the nearest Western Line station.

        Every station that falls in the walk band gets a walk edge.
        Every station that falls in the cab band (and is >= 1 km away) gets
        a cab edge.  A station < 1 km away gets a walk edge only (no cab).
        A station between 1–8 km gets a cab edge only (walking 1+ km to a
        station is unrealistic when a cab is available).
        """
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon
        all_items = list(self.data.station_coords.items())

        for ep, lat, lon in [("start", s_lat, s_lon), ("end", e_lat, e_lon)]:
            for sc, (st_lat, st_lon) in all_items:
                node = f"train_{sc}"
                if node not in self.G_multi:
                    continue

                h = haversine_km(lat, lon, st_lat, st_lon)

                if h <= WALK_RADIUS_KM:
                    if h <= SNAP_RADIUS_KM:
                        # ── Snap zone (≤200 m): zero-cost edge, no road graph ──
                        # Rail/metro stations have platform bridges so you can
                        # always cross to the correct exit — treat as free.
                        # Mark is_snap=True so the map renderer draws nothing.
                        if ep == "start":
                            self.G_multi.add_edge(
                                "start", node, weight=0, mode="walk",
                                distance_km=0, is_snap=True,
                                seg_start=(lat, lon), seg_end=(st_lat, st_lon))
                        else:
                            self.G_multi.add_edge(
                                node, "end", weight=0, mode="walk",
                                distance_km=0, is_snap=True,
                                seg_start=(st_lat, st_lon), seg_end=(lat, lon))
                    else:
                        # ── Walk zone: haversine * ROAD_FACTOR for walk time ──
                        d_km  = h * ROAD_FACTOR
                        walk_t = d_km * WALK_MIN_PER_KM
                        if ep == "start":
                            self.G_multi.add_edge(
                                "start", node, weight=walk_t, mode="walk",
                                distance_km=d_km,
                                seg_start=(lat, lon), seg_end=(st_lat, st_lon))
                        else:
                            self.G_multi.add_edge(
                                node, "end", weight=walk_t, mode="walk",
                                distance_km=d_km,
                                seg_start=(st_lat, st_lon), seg_end=(lat, lon))

                elif h <= CAB_RADIUS_KM:
                    # ── Cab zone: haversine x ROAD_FACTOR (no road-graph call) ──
                    d_km, cab_t = self._cab_dist_and_time(lat, lon, st_lat, st_lon)
                    if d_km < CAB_MIN_DIST_KM:
                        continue
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", node, weight=cab_t, mode="car",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(st_lat, st_lon))
                    else:
                        self.G_multi.add_edge(
                            node, "end", weight=cab_t, mode="car",
                            distance_km=d_km,
                            seg_start=(st_lat, st_lon), seg_end=(lat, lon))
                # else: beyond CAB_RADIUS_KM — skip

    # ── Bus layer ─────────────────────────────────────────────────────────

    def _get_relevant_routes(self) -> pd.DataFrame:
        """
        Include any bus route that has at least one stop within CAB_RADIUS_KM
        of the start OR end point.

        Using a radius (not a fixed count) keeps this consistent with the
        station endpoint logic: Dijkstra can consider any stop reachable by
        a cab from origin/destination, so we must load all routes that serve
        stops in that catchment area.
        """
        bus_df     = self.data.bus_df
        stops_uniq = (bus_df[["stop_id", "stop_lat", "stop_lon", "route_short_name"]]
                      .drop_duplicates("stop_id"))

        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        def routes_within_radius(lat, lon):
            mask = stops_uniq.apply(
                lambda r: haversine_km(lat, lon,
                                       float(r["stop_lat"]),
                                       float(r["stop_lon"])) <= CAB_RADIUS_KM,
                axis=1)
            return set(stops_uniq[mask]["route_short_name"])

        routes = (routes_within_radius(s_lat, s_lon) |
                  routes_within_radius(e_lat, e_lon))
        return bus_df[bus_df["route_short_name"].isin(routes)]

    def _build_bus_layer(self):
        """
        Build directed bus edges in ascending stop_sequence order only.

        Problem: Bus Dataset 3.csv stores both directions of a route under the
        same route_short_name, with stop_sequence restarting from 1 for each
        direction.  Naively sorting all rows by stop_sequence and iterating
        would mix the two directions (e.g. 1→5→2→3→4→6).

        Fix: split each route's rows into contiguous monotone-ascending runs
        (a "trip") by detecting where stop_sequence resets or does not increase.
        Each run is one direction and is added as a directed chain of edges.
        """
        G          = self.G_multi
        bus_subset = self._get_relevant_routes()

        for route, grp in bus_subset.groupby("route_short_name"):
            # Keep the original CSV order (do NOT sort globally)
            grp = grp.reset_index(drop=True)

            # Split into trips: start a new trip whenever stop_sequence is ≤
            # the previous stop_sequence (i.e. it resets or goes backward)
            trips = []
            current_trip = []
            last_seq = -1
            for _, row in grp.iterrows():
                seq = int(row["stop_sequence"])
                if seq <= last_seq:
                    # sequence reset → save completed trip, start new one
                    if len(current_trip) >= 2:
                        trips.append(current_trip)
                    current_trip = [row]
                else:
                    current_trip.append(row)
                last_seq = seq
            if len(current_trip) >= 2:
                trips.append(current_trip)

            # Add directed edges for each trip (ascending order only)
            for trip in trips:
                for i in range(1, len(trip)):
                    prev = trip[i - 1]
                    curr = trip[i]
                    n_prev = f"bus_{prev['stop_id']}"
                    n_curr = f"bus_{curr['stop_id']}"
                    G.add_node(n_prev,
                               lat=float(prev["stop_lat"]), lon=float(prev["stop_lon"]),
                               stop_name=prev["stop_name"])
                    G.add_node(n_curr,
                               lat=float(curr["stop_lat"]), lon=float(curr["stop_lon"]),
                               stop_name=curr["stop_name"])
                    d = haversine_km(float(prev["stop_lat"]), float(prev["stop_lon"]),
                                     float(curr["stop_lat"]), float(curr["stop_lon"]))
                    G.add_edge(n_prev, n_curr,
                               weight=d * BUS_MIN_PER_KM, mode="bus",
                               distance_km=d, route=route,
                               seg_start=(float(prev["stop_lat"]), float(prev["stop_lon"])),
                               seg_end  =(float(curr["stop_lat"]), float(curr["stop_lon"])))

        self._bus_subset = bus_subset

    def _connect_endpoints_to_bus(self):
        """
        Connect start/end to bus stops using the same search-radius model:

          Haversine <= WALK_RADIUS_KM -> walk edge
          Haversine <= CAB_RADIUS_KM  -> cab edge  (if >= CAB_MIN_DIST_KM)
        """
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        bus_nodes = [(n, a) for n, a in self.G_multi.nodes(data=True)
                     if n.startswith("bus_") and a.get("lat") is not None]

        for ep, lat, lon in [("start", s_lat, s_lon), ("end", e_lat, e_lon)]:
            for bn, ba in bus_nodes:
                h = haversine_km(lat, lon, ba["lat"], ba["lon"])

                if h <= WALK_RADIUS_KM:
                    # haversine * ROAD_FACTOR — no nearest_nodes call needed
                    d_km   = h * ROAD_FACTOR
                    walk_t = d_km * WALK_MIN_PER_KM
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", bn, weight=walk_t, mode="walk",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(ba["lat"], ba["lon"]))
                    else:
                        self.G_multi.add_edge(
                            bn, "end", weight=walk_t, mode="walk",
                            distance_km=d_km,
                            seg_start=(ba["lat"], ba["lon"]), seg_end=(lat, lon))

                elif h <= CAB_RADIUS_KM:
                    d_km, cab_t = self._cab_dist_and_time(lat, lon, ba["lat"], ba["lon"])
                    if d_km < CAB_MIN_DIST_KM:
                        continue
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", bn, weight=cab_t, mode="car",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(ba["lat"], ba["lon"]))
                    else:
                        self.G_multi.add_edge(
                            bn, "end", weight=cab_t, mode="car",
                            distance_km=d_km,
                            seg_start=(ba["lat"], ba["lon"]), seg_end=(lat, lon))

    # ── Transfers (combination) ───────────────────────────────────────────

    def _build_transfers(self, interchange_penalty=0.0):
        """
        Wire inter-layer transfer edges for multimodal modes.

        Bus <-> Train : walk edge when haversine <= MAX_TRANSFER_DIST_KM (0.5 km).
                        In least_interchange mode carries interchange_penalty.
        Cab           : transit<->transit within 20 km + direct start->end.

        Note: Train<->Train transfers at the same station are handled inside
        _build_train_layer so they exist even in single-rail-system builds.
        """
        G = self.G_multi

        train_nodes = [(n, a) for n, a in G.nodes(data=True)
                       if n.startswith("train_") and a.get("lat")]
        bus_nodes   = [(n, a) for n, a in G.nodes(data=True)
                       if n.startswith("bus_")   and a.get("lat")]

        # ── Bus <-> Train walk transfers ──────────────────────────────────
        # Use haversine directly (no road-graph call) — these pairs are
        # <= MAX_TRANSFER_DIST_KM (0.5 km) apart so haversine is accurate
        # enough, and avoids nearest_nodes which requires scikit-learn.
        for tn, ta in train_nodes:
            for bn, ba in bus_nodes:
                d_km = haversine_km(ta["lat"], ta["lon"], ba["lat"], ba["lon"])
                if d_km <= MAX_TRANSFER_DIST_KM:
                    t = d_km * WALK_MIN_PER_KM
                    w = t + interchange_penalty
                    for src, dst, sc, dc in [
                        (tn, bn, (ta["lat"], ta["lon"]), (ba["lat"], ba["lon"])),
                        (bn, tn, (ba["lat"], ba["lon"]), (ta["lat"], ta["lon"])),
                    ]:
                        G.add_edge(src, dst, weight=w, mode="walk",
                                   distance_km=d_km, seg_start=sc, seg_end=dc,
                                   is_transfer=True)

        # ── Cab edges: transit<->transit within 20 km ─────────────────────
        # Skipped entirely in public_transport mode (no cab allowed at all)
        if self.mode != "public_transport":
            all_transit = train_nodes + bus_nodes
            for i, (n_a, a_a) in enumerate(all_transit):
                for n_b, a_b in all_transit[i+1:]:
                    h = haversine_km(a_a["lat"], a_a["lon"], a_b["lat"], a_b["lon"])
                    if h > 20.0:
                        continue
                    d_km, cab_t = self._cab_dist_and_time(
                        a_a["lat"], a_a["lon"], a_b["lat"], a_b["lon"])
                    if d_km < CAB_MIN_DIST_KM:
                        continue
                    # No interchange_penalty on cab edges — taking a cab
                    # between transit nodes is not a "line change", it's just
                    # a cab ride. Adding penalty here caused Dijkstra to prefer
                    # exiting one stop early to get a shorter cab, even when
                    # riding to the correct station was faster overall.
                    for src, dst, sc, dc in [
                        (n_a, n_b, (a_a["lat"], a_a["lon"]), (a_b["lat"], a_b["lon"])),
                        (n_b, n_a, (a_b["lat"], a_b["lon"]), (a_a["lat"], a_a["lon"])),
                    ]:
                        G.add_edge(src, dst, weight=cab_t, mode="car",
                                   distance_km=d_km, seg_start=sc, seg_end=dc)

            # ── Direct start->end cab ─────────────────────────────────────
            s_lat, s_lon = self.start_latlon
            e_lat, e_lon = self.end_latlon
            d_km, cab_t = self._cab_dist_and_time(s_lat, s_lon, e_lat, e_lon)
            if d_km >= CAB_MIN_DIST_KM:
                G.add_edge("start", "end", weight=cab_t, mode="car",
                           distance_km=d_km,
                           seg_start=(s_lat, s_lon), seg_end=(e_lat, e_lon))

    def _connect_endpoints_walk_only(self):
        """
        Public-transport mode: connect start/end to ALL transit nodes within
        WALK_RADIUS_KM (1.5 km) by walking only — no cab edges.

        Rail/metro nodes within SNAP_RADIUS_KM (200 m) get a zero-cost snap
        edge (platform bridge logic, same as _connect_endpoints_to_train).
        Bus stops always use road-graph distance — no snap.
        """
        PUBLIC_WALK_RADIUS = 1.5
        s_lat, s_lon = self.start_latlon
        e_lat, e_lon = self.end_latlon

        transit_nodes = [
            (n, a) for n, a in self.G_multi.nodes(data=True)
            if (n.startswith("train_") or n.startswith("bus_"))
            and a.get("lat") is not None
        ]

        for ep, lat, lon in [("start", s_lat, s_lon), ("end", e_lat, e_lon)]:
            reachable = [
                (n, a) for n, a in transit_nodes
                if haversine_km(lat, lon, a["lat"], a["lon"]) <= PUBLIC_WALK_RADIUS
            ]

            if not reachable:
                reachable = sorted(
                    transit_nodes,
                    key=lambda x: haversine_km(lat, lon, x[1]["lat"], x[1]["lon"])
                )[:1]
                if reachable:
                    self.advisories.append(
                        f"No transit within {PUBLIC_WALK_RADIUS} km of "
                        f"{'start' if ep == 'start' else 'destination'} — "
                        f"using nearest stop (may require a longer walk).")

            for n, a in reachable:
                h_node  = haversine_km(lat, lon, a["lat"], a["lon"])
                is_rail = n.startswith("train_")

                if is_rail and h_node <= SNAP_RADIUS_KM:
                    # Rail/metro ≤200 m: zero-cost platform-bridge snap
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", n, weight=0, mode="walk",
                            distance_km=0, is_snap=True,
                            seg_start=(lat, lon), seg_end=(a["lat"], a["lon"]))
                    else:
                        self.G_multi.add_edge(
                            n, "end", weight=0, mode="walk",
                            distance_km=0, is_snap=True,
                            seg_start=(a["lat"], a["lon"]), seg_end=(lat, lon))
                else:
                    d_km, walk_t = self._road_dist_and_time(
                        lat, lon, a["lat"], a["lon"], WALK_MIN_PER_KM)
                    if d_km == float("inf"):
                        continue
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", n, weight=walk_t, mode="walk",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(a["lat"], a["lon"]))
                    else:
                        self.G_multi.add_edge(
                            n, "end", weight=walk_t, mode="walk",
                            distance_km=d_km,
                            seg_start=(a["lat"], a["lon"]), seg_end=(lat, lon))

    def _connect_endpoints_combination(self):
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
            return None, None, None, builder.advisories, G_multi

        path = nx.shortest_path(G_multi, "start", "end", weight="weight")

        # For least_interchange the weighted path length includes penalty
        # minutes that aren't real travel time — recompute true time by
        # summing only the edge base weights (stored in "weight" but we
        # retrieve them via _extract_steps which reads from edge attrs).
        steps = self._extract_steps(G_multi, path)

        # True travel time = sum of actual edge weights chosen by Dijkstra
        # (these already exclude penalty for display purposes — we record the
        # raw weight so the penalty correctly influences path selection, but
        # we show the user only realistic minutes).
        total_time = sum(s["time_min"] for s in steps)

        return path, steps, total_time, builder.advisories, G_multi

    def _extract_steps(self, G, path):
        """
        Extract per-step info from the shortest path.
        For transfer/interchange edges the stored 'weight' may include a
        penalty; we back it out so time_min reflects realistic travel time.
        """
        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = G[u][v]
            attr = min(edge_data.values(), key=lambda a: a.get("weight", float("inf")))

            raw_weight = attr.get("weight", 0)
            # If this is a transfer edge, subtract the interchange penalty so
            # the displayed time is realistic (the penalty only guides routing).
            if attr.get("is_transfer"):
                # Recover base time: for train transfers the base is
                # TRAIN_LINE_PENALTY; for bus↔train the base is the walk time.
                # We stored weight = base + interchange_penalty, but we don't
                # know the penalty here. Safest: cap displayed time at
                # TRAIN_LINE_PENALTY for zero-distance transfers, else keep it.
                if attr.get("distance_km", 0) == 0:
                    display_time = TRAIN_LINE_PENALTY
                else:
                    # walk transfer — just show the actual walk time (weight
                    # minus penalty, but penalty ≤ weight so min with base)
                    display_time = min(raw_weight, raw_weight - INTERCHANGE_PENALTY)
                    display_time = max(display_time, 0)
            else:
                display_time = raw_weight

            steps.append({
                "from":        u,
                "to":          v,
                "mode":        attr.get("mode", "?"),
                "distance_km": attr.get("distance_km", 0),
                "time_min":    display_time,
                "route":       attr.get("route", attr.get("line", "")),
                "seg_start":   attr.get("seg_start"),
                "seg_end":     attr.get("seg_end"),
                "is_snap":     attr.get("is_snap", False),
            })
        return steps


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZER (CLI / non-Streamlit use)
# ──────────────────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "walk":     "#aaaaaa",
    "train":    "#ff3333",   # local train — red
    "metro":    "#00e5c0",   # metro — teal/cyan
    "monorail": "#ff9800",   # monorail — orange
    "bus":      "#1e90ff",   # bus — blue
    "car":      "#ffd700",   # cab — gold
}

# Map System column values -> display mode key
SYSTEM_TO_MODE = {
    "Mumbai Local Train":    "train",
    "Mumbai Metro":          "metro",
    "Navi Mumbai Metro":     "metro",
    "Mumbai Monorail":       "monorail",
}


class Visualizer:

    def __init__(self, data: MumbaiData):
        self.data = data

    def plot(self, steps, start_coords, end_coords, title="Mumbai Multimodal Route"):
        G_road = self.data.G_road
        fig, ax = ox.plot_graph(G_road, show=False, close=False,
                                node_size=0, edge_color="#cccccc",
                                edge_linewidth=0.3, bgcolor="white")

        for step in steps:
            color = MODE_COLORS.get(step["mode"], "purple")
            seg_s = step.get("seg_start")
            seg_e = step.get("seg_end")
            if seg_s is None or seg_e is None:
                continue
            if step["mode"] == "train":
                ax.plot([seg_s[1], seg_e[1]], [seg_s[0], seg_e[0]],
                        color=color, linewidth=3, alpha=0.85, zorder=4)
            else:
                try:
                    n1 = ox.distance.nearest_nodes(G_road, seg_s[1], seg_s[0])
                    n2 = ox.distance.nearest_nodes(G_road, seg_e[1], seg_e[0])
                    rp = nx.shortest_path(G_road, n1, n2, weight="length")
                    lons = [G_road.nodes[n]["x"] for n in rp]
                    lats = [G_road.nodes[n]["y"] for n in rp]
                    ax.plot(lons, lats, color=color, linewidth=3, alpha=0.85, zorder=4)
                except Exception:
                    ax.plot([seg_s[1], seg_e[1]], [seg_s[0], seg_e[0]],
                            color=color, linewidth=2, linestyle="--", alpha=0.7, zorder=4)

        ax.scatter(start_coords[1], start_coords[0], c="#00e5c0", s=200, zorder=6, marker="*")
        ax.scatter(end_coords[1],   end_coords[0],   c="#ff3333", s=200, zorder=6, marker="*")
        ax.annotate("START", xy=(start_coords[1], start_coords[0]),
                    xytext=(5, 5), textcoords="offset points", color="#00e5c0", fontsize=9)
        ax.annotate("END",   xy=(end_coords[1],   end_coords[0]),
                    xytext=(5, 5), textcoords="offset points", color="#ff3333", fontsize=9)

        patches = [mpatches.Patch(color=c, label=m.title()) for m, c in MODE_COLORS.items()]
        patches += [mpatches.Patch(color="#00e5c0", label="Start"),
                    mpatches.Patch(color="#ff3333", label="End")]
        ax.legend(handles=patches, loc="upper left", fontsize=8)
        plt.title(title)
        plt.tight_layout()
        plt.savefig("route_map.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Map saved to route_map.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI MAIN
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_PLACES = {
    "Chhatrapati Shivaji Maharaj Terminus (CSMT)": (18.9398, 72.8354),
    "Dadar":        (19.0178, 72.8478),
    "Andheri":      (19.1196, 72.8470),
    "Bandra":       (19.0596, 72.8295),
    "Kurla":        (19.0728, 72.8826),
    "Thane":        (19.1972, 72.9780),
    "Borivali":     (19.2307, 72.8567),
    "Churchgate":   (18.9322, 72.8264),
    "Ghatkopar":    (19.0863, 72.9081),
    "Powai":        (19.1176, 72.9060),
    "BKC":          (19.0660, 72.8654),
    "Panvel":       (18.9894, 73.1175),
    "Mulund":       (19.1726, 72.9560),
    "Vashi":        (19.0771, 73.0071),
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
                return names[idx], KNOWN_PLACES[names[idx]]
        except ValueError:
            pass
        print("Invalid selection.")


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
        print("Invalid.")


def print_route(steps, total_time, advisories):
    print("\n" + "="*60)
    print("  ROUTE SUMMARY")
    print("="*60)
    mode_time = {}
    for step in steps:
        m = step["mode"]
        mode_time[m] = mode_time.get(m, 0) + step["time_min"]
        route_info = f"  [{step['route']}]" if step.get("route") else ""
        print(f"  {step['mode'].upper():6s}{route_info:25s}  "
              f"{step['distance_km']:5.2f} km  "
              f"{step['time_min']:5.1f} min  "
              f"{step['from']} → {step['to']}")
    print("-"*60)
    print(f"  TOTAL TIME: {total_time:.1f} minutes")
    for m, t in mode_time.items():
        print(f"    {m.upper():10s}: {t:.1f} min")
    if advisories:
        print("\n  Advisories:")
        for a in advisories:
            print(f"    • {a}")
    print("="*60)


def main():
    data   = MumbaiData()
    router = Router(data)
    viz    = Visualizer(data)
    start_name, start_coords = select_place("Select START:")
    end_name,   end_coords   = select_place("Select DESTINATION:")
    mode = select_mode()
    print(f"\nComputing {mode} route: {start_name} → {end_name} …")
    t0 = time.time()
    result = router.route(start_coords, end_coords, mode)
    print(f"Done in {time.time()-t0:.2f}s")
    if result[0] is None:
        print("No route found.")
        return
    path, steps, total_time, advisories, G_multi = result
    print_route(steps, total_time, advisories)
    viz.plot(steps, start_coords, end_coords,
             title=f"{mode.title()}: {start_name} → {end_name}")


if __name__ == "__main__":
    main()