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
  - Final Bus Dataset.csv
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
WALK_RADIUS_KM       = 1.5   # stations within this radius: walk edge (haversine)
CAB_RADIUS_KM        = 8.0   # stations within this radius: cab edge (haversine)
                              # Covers ~8 km — enough to reach a better line
                              # (e.g. Worli→Cotton Green 5 km, Worli→Byculla 4.5 km)
ROAD_FACTOR          = 1.3
MAX_TRANSFER_DIST_KM = 0.5
# Realistic wait times per destination mode (what you're waiting FOR after a transfer)
WAIT_TRAIN            = 5.0   # avg wait for local train
WAIT_METRO            = 5.0   # avg wait for metro
WAIT_MONORAIL         = 5.0   # avg wait for monorail
WAIT_BUS              = 15.0  # avg wait for bus
WAIT_CAB              = 2.0   # cab pickup time

ROUTING_NUDGE         = 1.0   # extra routing-only minutes to break ties in Dijkstra
                              # (not shown to user — only discourages unnecessary transfers)

def wait_for_mode(mode: str) -> float:
    """Return the realistic wait time for the given destination transit mode."""
    return {
        "train":    WAIT_TRAIN,
        "metro":    WAIT_METRO,
        "monorail": WAIT_MONORAIL,
        "bus":      WAIT_BUS,
        "car":      WAIT_CAB,
    }.get(mode, 0.0)
BUS_ROUTE_LIMIT      = 10
INTERCHANGE_PENALTY  = 15.0  # EXTRA minutes on top of BASE_TRANSFER_PENALTY in least_interchange mode

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

        elif self.mode in ("train", "metro", "bus",
                           "earliest_arrival", "least_interchange",
                           "public_transport"):
            # ── Compute routing-only transfer penalty ──────────────────
            # transfer_penalty is a routing-only nudge added to ALL transfer edges
            # so Dijkstra mildly discourages unnecessary changes.
            # Real wait time is per-mode (wait_for_mode) and IS shown to the user.
            # INTERCHANGE_PENALTY is an extra nudge only in least_interchange mode.
            transfer_penalty = ROUTING_NUDGE
            if self.mode == "least_interchange":
                transfer_penalty += INTERCHANGE_PENALTY

            # ── Build layers ──────────────────────────────────────────────
            if self.mode == "train":
                self._build_train_layer(allowed_systems=SYSTEMS_LOCAL_TRAIN,
                                        interchange_penalty=transfer_penalty)
                self._connect_endpoints_to_train()

            elif self.mode == "metro":
                self._build_train_layer(allowed_systems=SYSTEMS_METRO,
                                        interchange_penalty=transfer_penalty)
                self._connect_endpoints_to_train()

            elif self.mode == "bus":
                self._build_bus_layer()
                self._build_bus_transfers(interchange_penalty=transfer_penalty)
                self._connect_endpoints_to_bus()

            elif self.mode in ("earliest_arrival", "least_interchange"):
                self._build_train_layer(allowed_systems=SYSTEMS_ALL,
                                        interchange_penalty=transfer_penalty)
                self._build_bus_layer()
                self._build_transfers(interchange_penalty=transfer_penalty)
                self._build_bus_transfers(interchange_penalty=transfer_penalty)
                self._connect_endpoints_combination()

            elif self.mode == "public_transport":
                self._build_train_layer(allowed_systems=SYSTEMS_ALL,
                                        interchange_penalty=transfer_penalty)
                self._build_bus_layer()
                self._build_transfers(interchange_penalty=transfer_penalty)
                self._build_bus_transfers(interchange_penalty=transfer_penalty)
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
        interchange_penalty : total penalty on transfer edges
                      (wait_for_mode(dst_mode) + ROUTING_NUDGE [+ INTERCHANGE_PENALTY]).
                      Added to
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

                # Resolve display mode from System column
                sys_val   = str(row_i.get("System", ""))
                edge_mode = SYSTEM_TO_MODE.get(sys_val, "train")

                G.add_node(node_i,
                           lat=coords_i[0], lon=coords_i[1],
                           station=row_i["Station Name"], line=line,
                           system=sys_val, edge_mode=edge_mode)
                G.add_node(node_i1,
                           lat=coords_i1[0], lon=coords_i1[1],
                           station=row_i1["Station Name"], line=line,
                           system=str(row_i1.get("System", "")),
                           edge_mode=SYSTEM_TO_MODE.get(str(row_i1.get("System", "")), "train"))

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
                    # interchange_penalty is routing-only; wait_for_mode gives
                    # the realistic wait time for the destination transit mode.
                    wt_i = wait_for_mode(ai.get("edge_mode", edge_mode))
                    wt_j = wait_for_mode(aj.get("edge_mode", edge_mode))
                    w_ij = wt_i + interchange_penalty   # ni→nj: waiting for nj's line
                    w_ji = wt_j + interchange_penalty   # nj→ni: waiting for ni's line
                    G.add_edge(ni, nj, weight=w_ij, mode="walk",
                               distance_km=0, seg_start=ci, seg_end=cj,
                               is_transfer=True, base_time=0, wait_time=wt_i)
                    G.add_edge(nj, ni, weight=w_ji, mode="walk",
                               distance_km=0, seg_start=cj, seg_end=ci,
                               is_transfer=True, base_time=0, wait_time=wt_j)

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
                    # Resolve what mode this train node runs on (train/metro/monorail)
                    node_edge_mode = self.G_multi.nodes[node].get("edge_mode", "train")                                      if node in self.G_multi else "train"
                    boarding_wait  = wait_for_mode(node_edge_mode)

                    if h <= SNAP_RADIUS_KM:
                        # ── Snap zone (≤200 m): zero walk cost, but still need
                        #    to wait on the platform for the next service.
                        # Mark is_snap=True so the map renderer draws nothing.
                        if ep == "start":
                            self.G_multi.add_edge(
                                "start", node,
                                weight=boarding_wait, mode="walk",
                                distance_km=0, is_snap=True,
                                is_transfer=True, base_time=0,
                                wait_time=boarding_wait,
                                seg_start=(lat, lon), seg_end=(st_lat, st_lon))
                        else:
                            self.G_multi.add_edge(
                                node, "end", weight=0, mode="walk",
                                distance_km=0, is_snap=True,
                                seg_start=(st_lat, st_lon), seg_end=(lat, lon))
                    else:
                        # ── Walk zone: haversine * ROAD_FACTOR for walk time ──
                        d_km   = h * ROAD_FACTOR
                        walk_t = d_km * WALK_MIN_PER_KM
                        if ep == "start":
                            # Walk to station + wait for the service
                            self.G_multi.add_edge(
                                "start", node,
                                weight=walk_t + boarding_wait, mode="walk",
                                distance_km=d_km,
                                seg_start=(lat, lon), seg_end=(st_lat, st_lon),
                                is_transfer=True, base_time=walk_t,
                                wait_time=boarding_wait)
                        else:
                            self.G_multi.add_edge(
                                node, "end", weight=walk_t, mode="walk",
                                distance_km=d_km,
                                seg_start=(st_lat, st_lon), seg_end=(lat, lon))

                elif h <= CAB_RADIUS_KM and self.mode not in ("bus", "train", "metro", "public_transport"):
                    # ── Cab zone: haversine x ROAD_FACTOR (no road-graph call) ──
                    d_km, cab_t = self._cab_dist_and_time(lat, lon, st_lat, st_lon)
                    if d_km < CAB_MIN_DIST_KM:
                        continue
                    if ep == "start":
                        node_mode = self.G_multi.nodes[node].get("edge_mode", "train")                                     if node in self.G_multi else "train"
                        bwt = wait_for_mode(node_mode)
                        self.G_multi.add_edge(
                            "start", node, weight=cab_t + bwt, mode="car",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(st_lat, st_lon),
                            wait_time=bwt, base_time=cab_t)
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

        Problem: Final Bus Dataset.csv stores both directions of a route under the
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
                    # Include route in node ID so the same physical stop
                    # on different routes is a distinct graph node.
                    # This forces Dijkstra to use an explicit transfer edge to
                    # switch routes — it cannot silently hop between routes.
                    n_prev = f"bus_{prev['stop_id']}__{route}"
                    n_curr = f"bus_{curr['stop_id']}__{route}"
                    G.add_node(n_prev,
                               lat=float(prev["stop_lat"]), lon=float(prev["stop_lon"]),
                               stop_name=prev["stop_name"], route=route,
                               stop_id=str(prev["stop_id"]))
                    G.add_node(n_curr,
                               lat=float(curr["stop_lat"]), lon=float(curr["stop_lon"]),
                               stop_name=curr["stop_name"], route=route,
                               stop_id=str(curr["stop_id"]))
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
                        # Add boarding wait for the bus you're about to take
                        self.G_multi.add_edge(
                            "start", bn,
                            weight=walk_t + WAIT_BUS,
                            mode="walk",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(ba["lat"], ba["lon"]),
                            is_transfer=True, base_time=walk_t, wait_time=WAIT_BUS)
                    else:
                        self.G_multi.add_edge(
                            bn, "end", weight=walk_t, mode="walk",
                            distance_km=d_km,
                            seg_start=(ba["lat"], ba["lon"]), seg_end=(lat, lon))

                elif h <= CAB_RADIUS_KM and self.mode not in ("bus", "train", "metro", "public_transport"):
                    d_km, cab_t = self._cab_dist_and_time(lat, lon, ba["lat"], ba["lon"])
                    if d_km < CAB_MIN_DIST_KM:
                        continue
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", bn, weight=cab_t + WAIT_BUS, mode="car",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(ba["lat"], ba["lon"]),
                            wait_time=WAIT_BUS, base_time=cab_t)
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
                    wt_to_bus   = WAIT_BUS
                    wt_to_train = wait_for_mode(ta.get("edge_mode", "train"))
                    for src, dst, sc, dc, wt in [
                        (tn, bn, (ta["lat"], ta["lon"]), (ba["lat"], ba["lon"]), wt_to_bus),
                        (bn, tn, (ba["lat"], ba["lon"]), (ta["lat"], ta["lon"]), wt_to_train),
                    ]:
                        G.add_edge(src, dst, weight=t + wt + interchange_penalty,
                                   mode="walk",
                                   distance_km=d_km, seg_start=sc, seg_end=dc,
                                   is_transfer=True, base_time=t, wait_time=wt)

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
                    # Add boarding wait for the destination transit node.
                    # When you arrive at a station by cab you still wait for
                    # the next service, so this must be in Dijkstra's weight.
                    wt_ab = wait_for_mode(a_b.get("edge_mode", "train")
                                          if n_b.startswith("train_") else "bus")
                    wt_ba = wait_for_mode(a_a.get("edge_mode", "train")
                                          if n_a.startswith("train_") else "bus")
                    G.add_edge(n_a, n_b, weight=cab_t + wt_ab, mode="car",
                               distance_km=d_km,
                               seg_start=(a_a["lat"], a_a["lon"]),
                               seg_end=(a_b["lat"], a_b["lon"]),
                               wait_time=wt_ab, base_time=cab_t)
                    G.add_edge(n_b, n_a, weight=cab_t + wt_ba, mode="car",
                               distance_km=d_km,
                               seg_start=(a_b["lat"], a_b["lon"]),
                               seg_end=(a_a["lat"], a_a["lon"]),
                               wait_time=wt_ba, base_time=cab_t)

            # ── Direct start->end cab ─────────────────────────────────────
            s_lat, s_lon = self.start_latlon
            e_lat, e_lon = self.end_latlon
            d_km, cab_t = self._cab_dist_and_time(s_lat, s_lon, e_lat, e_lon)
            if d_km >= CAB_MIN_DIST_KM:
                G.add_edge("start", "end", weight=cab_t, mode="car",
                           distance_km=d_km,
                           seg_start=(s_lat, s_lon), seg_end=(e_lat, e_lon))

    def _build_bus_transfers(self, interchange_penalty=0.0):
        """
        Add walk transfer edges between bus stops of DIFFERENT routes that are
        within MAX_TRANSFER_DIST_KM (0.5 km) of each other.

        This enables bus-to-bus interchanges.  Without this, changing bus routes
        is impossible mid-journey (the only path would be bus→walk→train→bus
        which is always longer and goes through the bus↔train penalty).

        interchange_penalty is applied the same way as bus↔train transfers:
        weight = walk_time + interchange_penalty, base_time = walk_time stored
        separately so _extract_steps can display realistic times.

        Only runs if bus layer has been built (_bus_subset exists).
        """
        if not hasattr(self, "_bus_subset"):
            return

        G = self.G_multi
        bus_nodes = [(n, a) for n, a in G.nodes(data=True)
                     if n.startswith("bus_") and a.get("lat") is not None]

        # Since node IDs are now bus_{stop_id}__{route}, two nodes are on the
        # same route iff they share the same route suffix. Extract route from node name.
        def node_route(n):
            parts = n.split("__", 1)
            return parts[1] if len(parts) == 2 else ""

        for i, (n_a, a_a) in enumerate(bus_nodes):
            for n_b, a_b in bus_nodes[i+1:]:
                d_km = haversine_km(a_a["lat"], a_a["lon"], a_b["lat"], a_b["lon"])
                if d_km > MAX_TRANSFER_DIST_KM:
                    continue
                # Skip if same route — no transfer needed within the same route
                if node_route(n_a) == node_route(n_b):
                    continue
                t = d_km * WALK_MIN_PER_KM
                wt = WAIT_BUS   # both directions: waiting for a bus
                for src, dst, sc, dc in [
                    (n_a, n_b, (a_a["lat"], a_a["lon"]), (a_b["lat"], a_b["lon"])),
                    (n_b, n_a, (a_b["lat"], a_b["lon"]), (a_a["lat"], a_a["lon"])),
                ]:
                    G.add_edge(src, dst, weight=t + wt + interchange_penalty,
                               mode="walk",
                               distance_km=d_km, seg_start=sc, seg_end=dc,
                               is_transfer=True, base_time=t, wait_time=wt)

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

                node_mode = a.get("edge_mode", "bus" if n.startswith("bus_") else "train")
                bwt = wait_for_mode(node_mode)

                if is_rail and h_node <= SNAP_RADIUS_KM:
                    # Rail/metro ≤200 m: snap walk but still wait for service
                    if ep == "start":
                        self.G_multi.add_edge(
                            "start", n, weight=bwt, mode="walk",
                            distance_km=0, is_snap=True,
                            is_transfer=True, base_time=0, wait_time=bwt,
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
                            "start", n, weight=walk_t + bwt, mode="walk",
                            distance_km=d_km,
                            seg_start=(lat, lon), seg_end=(a["lat"], a["lon"]),
                            is_transfer=True, base_time=walk_t, wait_time=bwt)
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

    TRANSIT_MODES = {"train", "metro", "monorail", "bus"}

    def _extract_steps(self, G, path):
        """
        Extract per-step info from the shortest path.

        Wait time injection:
        - For explicit transfer edges (is_transfer=True): display_time =
          base_time (walk) + wait_time stored on the edge. The routing nudge
          is stripped.
        - For implicit boardings (two consecutive transit edges whose route/line
          changes, with no transfer edge between them — e.g. two bus routes
          sharing a stop node): inject a synthetic wait step immediately before
          the new boarding.  wait_for_mode(new_mode) gives the wait duration.
        - First boarding from "start" counts as an implicit boarding too — you
          always wait for your first vehicle.
        """
        TRANSIT = self.TRANSIT_MODES

        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = G[u][v]
            attr = min(edge_data.values(), key=lambda a: a.get("weight", float("inf")))

            wt = attr.get("wait_time", 0)

            if attr.get("is_transfer"):
                is_endpoint_edge = (u == "start" or v == "end")
                base_time = attr.get("base_time", 0)

                if is_endpoint_edge and wt > 0:
                    # Split: walk card (base_time) + wait card (wt) for start→transit.
                    # Only emit walk card if there's actually walking (base_time > 0).
                    if base_time > 0 and not attr.get("is_snap"):
                        steps.append({
                            "from":        u,
                            "to":          v,
                            "mode":        attr.get("mode", "walk"),
                            "distance_km": attr.get("distance_km", 0),
                            "time_min":    base_time,
                            "route":       "",
                            "seg_start":   attr.get("seg_start"),
                            "seg_end":     attr.get("seg_end"),
                            "is_snap":     False,
                            "is_transfer": False,
                        })
                    # Wait card — boarding wait at the station
                    if u != "end" and v != "end":
                        steps.append({
                            "from":        v,
                            "to":          v,
                            "mode":        "wait",
                            "distance_km": 0,
                            "time_min":    wt,
                            "route":       "",
                            "seg_start":   attr.get("seg_end"),
                            "seg_end":     attr.get("seg_end"),
                            "is_snap":     False,
                            "is_transfer": True,
                        })
                else:
                    # Mid-route transfer or no wait: single step rendered as "wait"
                    steps.append({
                        "from":        u,
                        "to":          v,
                        "mode":        "wait" if not is_endpoint_edge else attr.get("mode", "?"),
                        "distance_km": attr.get("distance_km", 0),
                        "time_min":    base_time + wt,
                        "route":       attr.get("route", attr.get("line", "")),
                        "seg_start":   attr.get("seg_start"),
                        "seg_end":     attr.get("seg_end"),
                        "is_snap":     attr.get("is_snap", False),
                        "is_transfer": attr.get("is_transfer", False),
                    })

            elif attr.get("mode") == "car" and wt > 0:
                # Cab edge with a boarding wait baked in:
                # emit the cab travel as one step, then a separate "wait" step.
                # This way the wait shows as its own ⏳ card rather than being
                # silently absorbed into the cab time.
                base_cab = attr.get("base_time", attr.get("weight", 0) - wt)
                steps.append({
                    "from":        u,
                    "to":          v,
                    "mode":        "car",
                    "distance_km": attr.get("distance_km", 0),
                    "time_min":    base_cab,
                    "route":       "",
                    "seg_start":   attr.get("seg_start"),
                    "seg_end":     attr.get("seg_end"),
                    "is_snap":     False,
                    "is_transfer": False,
                })
                # Only add wait step if this is a boarding (destination is transit),
                # not if it's the last-mile to "end".
                if v != "end":
                    steps.append({
                        "from":        v,
                        "to":          v,
                        "mode":        "wait",
                        "distance_km": 0,
                        "time_min":    wt,
                        "route":       "",
                        "seg_start":   attr.get("seg_end"),
                        "seg_end":     attr.get("seg_end"),
                        "is_snap":     False,
                        "is_transfer": True,
                    })

            else:
                steps.append({
                    "from":        u,
                    "to":          v,
                    "mode":        attr.get("mode", "?"),
                    "distance_km": attr.get("distance_km", 0),
                    "time_min":    attr.get("weight", 0),
                    "route":       attr.get("route", attr.get("line", "")),
                    "seg_start":   attr.get("seg_start"),
                    "seg_end":     attr.get("seg_end"),
                    "is_snap":     attr.get("is_snap", False),
                    "is_transfer": False,
                })

        # Wait times are baked into graph weights so Dijkstra compared routes
        # fairly. Here we split cab+wait into separate display steps so the
        # user sees the wait as its own ⏳ card.
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