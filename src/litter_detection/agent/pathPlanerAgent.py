"""PathPlannerAgent — Frontier-based coverage exploration with LiDAR obstacle avoidance.

Ablauf pro Iteration (NEXT_FRONTIER-Request):
  1. Aktuelle Roboterpose per Zenoh holen
  2. Kamera-Footprint an aktueller Position als "explored" markieren
  3. Frontier-Zellen finden (Grenze zwischen explored und unknown)
  4. Nächste erreichbare Frontier per A* ansteuern
  5. A*-Pfad als Wegpunktliste zurückgeben
  → kein Frontier mehr: status="completed"

Die Belegungskarte (Hindernisse) wird laufend aus LiDAR-Scans gebaut.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import threading
import time

import numpy as np
import zenoh

from litter_detection.agent.models import OdometryState

logger = logging.getLogger("pathplanner")

# --- Planner constants ---
OBSTACLE_Z_MIN_M = 0.05   # points below this are ground (free)
OBSTACLE_Z_MAX_M = 0.80   # points above this are overhead clearance (free)
GRID_CELL_SIZE_M = 0.20   # occupancy/frontier grid resolution
INFLATE_CELLS = 2          # obstacle inflation radius in cells (~0.4 m safety margin)
LIDAR_WARMUP_S = 2.0       # max seconds to wait for first LiDAR scan before planning
FRONTIER_SIZE_WEIGHT = 3.0 # scales cluster-size bonus in the frontier cost function

_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS_8 = _NEIGHBORS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------------------------------------------------------------------------
# Occupancy + explored grid
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """2D grid that tracks obstacles (from LiDAR) and explored area (from camera footprint).

    Three logical cell states:
      - occupied  : LiDAR detected an obstacle (inflated for safety)
      - explored  : camera has seen this cell from a visited pose
      - unknown   : everything else (target for frontier search)

    Frontiers are explored cells whose 4-neighbours contain at least one unknown cell.
    """

    def __init__(
        self,
        cell_size_m: float = GRID_CELL_SIZE_M,
        inflate_cells: int = INFLATE_CELLS,
    ) -> None:
        self.cell_size = cell_size_m
        self._inflate = inflate_cells
        self._raw_occupied: set[tuple[int, int]] = set()
        self._inflated: set[tuple[int, int]] = set()
        self._explored: set[tuple[int, int]] = set()
        self._dirty = False
        self._lock = threading.Lock()

    # --- obstacle ingestion ---

    def update_from_points(self, points: np.ndarray) -> None:
        """Project 3-D LiDAR points at obstacle height into 2-D grid cells."""
        mask = (points[:, 2] >= OBSTACLE_Z_MIN_M) & (points[:, 2] <= OBSTACLE_Z_MAX_M)
        new_cells: set[tuple[int, int]] = {
            (int(math.floor(p[0] / self.cell_size)),
             int(math.floor(p[1] / self.cell_size)))
            for p in points[mask]
        }
        with self._lock:
            before = len(self._raw_occupied)
            self._raw_occupied.update(new_cells)
            if len(self._raw_occupied) > before:
                self._dirty = True

    def _rebuild_inflated(self) -> None:
        r = self._inflate
        inflated: set[tuple[int, int]] = set()
        for cx, cy in self._raw_occupied:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    inflated.add((cx + dx, cy + dy))
        self._inflated = inflated
        self._dirty = False

    def snapshot(self) -> set[tuple[int, int]]:
        """Thread-safe copy of the inflated obstacle set (for A*)."""
        with self._lock:
            if self._dirty:
                self._rebuild_inflated()
            return set(self._inflated)

    # --- explored area ---

    def mark_camera_footprint(
        self,
        world_x: float,
        world_y: float,
        half_w: float,
        half_l: float,
    ) -> None:
        """Mark the rectangular camera footprint centred at (world_x, world_y) as explored."""
        x_lo = int(math.floor((world_x - half_w) / self.cell_size))
        x_hi = int(math.ceil((world_x + half_w) / self.cell_size))
        y_lo = int(math.floor((world_y - half_l) / self.cell_size))
        y_hi = int(math.ceil((world_y + half_l) / self.cell_size))
        with self._lock:
            for cx in range(x_lo, x_hi + 1):
                for cy in range(y_lo, y_hi + 1):
                    self._explored.add((cx, cy))

    def get_frontiers(
        self,
        x_bounds: tuple[int, int],
        y_bounds: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Return unknown cells directly adjacent to explored area within field bounds."""
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        frontiers: set[tuple[int, int]] = set()
        with self._lock:
            explored = set(self._explored)
        for cx, cy in explored:
            for dx, dy in _NEIGHBORS_4:
                nb = (cx + dx, cy + dy)
                if not (x_min <= nb[0] <= x_max and y_min <= nb[1] <= y_max):
                    continue
                if nb not in explored and nb not in occupied:
                    frontiers.add(nb)
        return frontiers

    # --- coordinate helpers ---

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def grid_to_world(self, cx: int, cy: int) -> tuple[float, float]:
        half = self.cell_size / 2.0
        return (cx * self.cell_size + half, cy * self.cell_size + half)


# ---------------------------------------------------------------------------
# A* pathfinding helpers
# ---------------------------------------------------------------------------

def _astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    occupied: set[tuple[int, int]],
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """A* on a 2-D 8-connected grid; returns path (start…goal) or None."""
    if start == goal:
        return [start]

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    def h(c: tuple[int, int]) -> float:
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    heap: list[tuple[float, tuple[int, int]]] = [(h(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g: dict[tuple[int, int], float] = {start: 0.0}

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path: list[tuple[int, int]] = []
            node = cur
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return list(reversed(path))

        cx, cy = cur
        for dx, dy in _NEIGHBORS_8:
            nb = (cx + dx, cy + dy)
            if not (x_min <= nb[0] <= x_max and y_min <= nb[1] <= y_max):
                continue
            if nb in occupied:
                continue
            ng = g[cur] + math.hypot(dx, dy)
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                heapq.heappush(heap, (ng + h(nb), nb))
                came_from[nb] = cur

    return None


def _nearest_free(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    max_r: int = 5,
) -> tuple[int, int]:
    """Return cell itself or nearest non-occupied cell within max_r steps."""
    if cell not in occupied:
        return cell
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                nb = (cell[0] + dx, cell[1] + dy)
                if not (x_min <= nb[0] <= x_max and y_min <= nb[1] <= y_max):
                    continue
                if nb not in occupied:
                    return nb
    return cell


def _thin_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove collinear intermediate cells to reduce waypoint count."""
    if len(path) <= 2:
        return path
    result = [path[0]]
    for i in range(1, len(path) - 1):
        dx1, dy1 = path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]
        dx2, dy2 = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            result.append(path[i])
    result.append(path[-1])
    return result


def _cluster_frontiers(
    frontiers: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    """Group frontier cells into connected clusters using 4-connectivity."""
    remaining = set(frontiers)
    clusters: list[set[tuple[int, int]]] = []
    while remaining:
        seed = next(iter(remaining))
        cluster: set[tuple[int, int]] = set()
        queue = [seed]
        while queue:
            cell = queue.pop()
            if cell in remaining:
                remaining.discard(cell)
                cluster.add(cell)
                cx, cy = cell
                for dx, dy in _NEIGHBORS_4:
                    nb = (cx + dx, cy + dy)
                    if nb in remaining:
                        queue.append(nb)
        clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Zenoh clients
# ---------------------------------------------------------------------------

class ZenohLidarClient:
    """Subscribes to the LiDAR point-cloud topic and feeds an OccupancyGrid."""

    def __init__(
        self,
        topic: str,
        router: str | None,
        grid: OccupancyGrid,
        warmup_s: float = LIDAR_WARMUP_S,
    ) -> None:
        self._grid = grid
        self._warmup_s = warmup_s
        self._received = 0
        self._lock = threading.Lock()

        conf = zenoh.Config()
        if router:
            conf.insert_json5("connect/endpoints", json.dumps([router]))
        logger.info("ZenohLidarClient: opening session → %s (topic=%s)", router, topic)
        self._session = zenoh.open(conf)
        self._sub = self._session.declare_subscriber(topic, self._on_lidar)
        logger.info("ZenohLidarClient: subscribed to %s", topic)

    def _on_lidar(self, sample: zenoh.Sample) -> None:
        raw = json.loads(bytes(sample.payload).decode("utf-8"))
        # Sim: {"points": [[x,y,z], ...]}
        # Go2 (rt/utlidar/voxel_map_compressed): {"data": {"data": {"points": [...]}}}
        pts = raw.get("points")
        if pts is None:
            pts = raw.get("data", {}).get("data", {}).get("points", [])
        if not pts:
            return
        positions = np.array(pts, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            return
        self._grid.update_from_points(positions)
        with self._lock:
            self._received += 1

    def wait_for_scan(self) -> None:
        """Block until at least one scan is received or warmup time elapses."""
        deadline = time.time() + self._warmup_s
        while time.time() < deadline:
            with self._lock:
                if self._received > 0:
                    return
            time.sleep(0.05)

    def close(self) -> None:
        self._sub.undeclare()
        self._session.close()


class ZenohRobotLocalizationClient:
    """Subscribes to the odometry topic and provides the current robot pose."""

    def __init__(
        self,
        topic: str = "robodog/system_state/odometry",
        router: str | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        self.topic = topic
        self.timeout_s = timeout_s
        self.latest_pose: dict | None = None

        conf = zenoh.Config()
        if router:
            conf.insert_json5("connect/endpoints", json.dumps([router]))
        logger.info("ZenohRobotLocalizationClient: opening session → %s (topic=%s)", router, topic)
        self._session = zenoh.open(conf)
        self._sub = self._session.declare_subscriber(topic, self._on_pose)
        logger.info("ZenohRobotLocalizationClient: subscribed to %s", topic)

    def _on_pose(self, sample: zenoh.Sample) -> None:
        raw = json.loads(bytes(sample.payload).decode("utf-8"))
        state = OdometryState.from_raw(raw)
        if state is None:
            return
        qx, qy, qz, qw = state.quaternion
        yaw_rad = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        self.latest_pose = {
            "x": state.x,
            "y": state.y,
            "theta_deg": math.degrees(yaw_rad) % 360.0,
        }

    def get_current_pose(self) -> dict:
        deadline = time.time() + self.timeout_s
        while self.latest_pose is None:
            if time.time() > deadline:
                raise TimeoutError(
                    f"No robot pose received on Zenoh topic '{self.topic}'"
                )
            time.sleep(0.05)
        return self.latest_pose


# ---------------------------------------------------------------------------
# PathPlannerAgent
# ---------------------------------------------------------------------------

class PathPlannerAgent:
    """Frontier-based path planner.

    Jeder NEXT_FRONTIER-Request:
      - markiert die aktuelle Kamera-Footprint als "explored"
      - sucht die nächste erreichbare Frontier-Zelle
      - gibt den A*-Pfad dorthin als Wegpunktliste zurück
    """

    def __init__(
        self,
        localization_client=None,
        lidar_client=None,
        grid_cell_size_m: float = GRID_CELL_SIZE_M,
        lidar_warmup_s: float = LIDAR_WARMUP_S,
    ) -> None:
        from litter_detection.config import Settings
        cfg = Settings()

        logger.info("PathPlannerAgent: connecting to odometry topic …")
        self.localization_client = localization_client or ZenohRobotLocalizationClient(
            topic=cfg.topic_odometry,
            router=cfg.ZENOH_ROUTER,
        )

        self._grid = OccupancyGrid(cell_size_m=grid_cell_size_m)

        # Camera footprint half-extents derived from mounting geometry
        h = cfg.CAMERA_HEIGHT_M
        self._footprint_half_w = h * math.tan(math.radians(cfg.CAMERA_FOV_H_DEG / 2.0))
        self._footprint_half_l = h * math.tan(math.radians(cfg.CAMERA_FOV_V_DEG / 2.0))

        # Field origin stored on first NEXT_FRONTIER call
        self._field_origin: tuple[float, float] | None = None

        if lidar_client is None:
            logger.info("PathPlannerAgent: connecting to LiDAR topic …")
            self._lidar_client: ZenohLidarClient | None = ZenohLidarClient(
                topic=cfg.topic_lidar,
                router=cfg.ZENOH_ROUTER,
                grid=self._grid,
                warmup_s=lidar_warmup_s,
            )
        else:
            # Pass lidar_client=False to disable LiDAR (e.g. unit tests)
            self._lidar_client = lidar_client if lidar_client is not False else None

        logger.info("PathPlannerAgent: ready (cell_size=%.2fm, footprint=%.2fx%.2fm)",
                    grid_cell_size_m, self._footprint_half_w * 2, self._footprint_half_l * 2)

    # --- public API ---

    def is_path_valid(self, waypoints: list[dict]) -> bool:
        """Return False if any remaining waypoint cell is now inside an inflated obstacle."""
        occupied = self._grid.snapshot()
        for wp in waypoints:
            cell = self._grid.world_to_grid(wp["x"], wp["y"])
            if cell in occupied:
                return False
        return True

    def handle_request(self, request: dict) -> dict:
        if request.get("type") == "NEXT_FRONTIER":
            return self._next_frontier(request)
        return {"status": "error", "agent": "pathplanner", "message": "Unknown request type"}

    # --- frontier logic ---

    def _next_frontier(self, request: dict) -> dict:
        if "field_size" not in request:
            return {"status": "error", "agent": "pathplanner", "message": "Missing field_size"}

        field_size = request["field_size"]
        width = field_size["width_m"]
        height = field_size["height_m"]

        if width <= 0 or height <= 0:
            return {"status": "error", "agent": "pathplanner",
                    "message": "field_size must have positive width and height"}

        try:
            pose = self.localization_client.get_current_pose()
        except Exception as error:
            return {"status": "error", "agent": "pathplanner",
                    "message": f"Could not get robot pose: {error}"}

        # Store field origin once (robot position at first call = field corner)
        if self._field_origin is None:
            self._field_origin = (pose["x"], pose["y"])

        origin_x, origin_y = self._field_origin

        # Mark current camera footprint as explored
        self._grid.mark_camera_footprint(
            pose["x"], pose["y"],
            self._footprint_half_w, self._footprint_half_l,
        )

        # Block until at least one LiDAR scan has been processed
        if self._lidar_client is not None:
            self._lidar_client.wait_for_scan()

        occupied = self._grid.snapshot()

        # Grid bounds (field + 2-cell margin)
        cell = self._grid.cell_size
        x_bounds = (
            int(math.floor(origin_x / cell)) - 2,
            int(math.ceil((origin_x + width) / cell)) + 2,
        )
        y_bounds = (
            int(math.floor(origin_y / cell)) - 2,
            int(math.ceil((origin_y + height) / cell)) + 2,
        )

        frontiers = self._grid.get_frontiers(x_bounds, y_bounds, occupied)

        if not frontiers:
            logger.info("PathPlannerAgent: no frontiers remaining — exploration complete")
            return {"status": "completed", "agent": "pathplanner"}

        robot_cell = _nearest_free(
            self._grid.world_to_grid(pose["x"], pose["y"]),
            occupied, x_bounds, y_bounds,
        )

        # Cluster frontiers and score each cluster by size and distance.
        # cost = distance_to_nearest_cell / log(1 + cluster_size * FRONTIER_SIZE_WEIGHT)
        # Lower cost = better: prefer large, nearby frontier clusters.
        clusters = _cluster_frontiers(frontiers)
        scored: list[tuple[float, tuple[int, int]]] = []
        for cluster in clusters:
            nearest = min(
                cluster,
                key=lambda c: math.hypot(c[0] - robot_cell[0], c[1] - robot_cell[1]),
            )
            dist = math.hypot(nearest[0] - robot_cell[0], nearest[1] - robot_cell[1])
            size_bonus = math.log(1.0 + len(cluster) * FRONTIER_SIZE_WEIGHT)
            scored.append((dist / size_bonus, nearest))
        scored.sort()

        path: list[tuple[int, int]] | None = None
        target_cell: tuple[int, int] | None = None
        for _cost, candidate in scored:
            p = _astar(robot_cell, candidate, occupied, x_bounds, y_bounds)
            if p is not None:
                path = p
                target_cell = candidate
                break

        if path is None or target_cell is None:
            logger.info("PathPlannerAgent: all frontiers unreachable — exploration complete")
            return {"status": "completed", "agent": "pathplanner"}

        # Remove collinear intermediate points to reduce navigator calls
        thinned = _thin_path(path)
        waypoints = []
        for i, gc in enumerate(thinned[1:]):
            wx, wy = self._grid.grid_to_world(*gc)
            waypoints.append({"id": f"wp_{i}", "x": round(wx, 3), "y": round(wy, 3)})

        tx, ty = self._grid.grid_to_world(*target_cell)
        logger.info(
            "PathPlannerAgent: %d frontiers → target (%.2f, %.2f), %d waypoints",
            len(frontiers), tx, ty, len(waypoints),
        )

        return {
            "status": "success",
            "agent": "pathplanner",
            "waypoints": waypoints,
            "frontier": {"x": round(tx, 3), "y": round(ty, 3)},
        }
