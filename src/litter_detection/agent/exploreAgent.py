import logging
import threading

from litter_detection.agent.pathPlanerAgent import PathPlannerAgent

logger = logging.getLogger("explore-agent")
from litter_detection.agent.navigator import PointNavigator
from litter_detection.agent.models import MovementCommand, MovementSource
from litter_detection.agent.tools.motion_types import RobotMotionGateway
from litter_detection.config import Settings
from litter_detection.zenoh_session import open_zenoh_session

WAYPOINT_TIMEOUT_S = 10.0  # seconds before giving up on a single waypoint and re-planning


def _build_gateway() -> RobotMotionGateway:
    cfg = Settings()
    return RobotMotionGateway(
        router=cfg.ZENOH_ROUTER,
        movement_topic=cfg.topic_movement_command,
        dry_run=False,
    )


class ExploreAgent:
    def __init__(self, pathplanner=None, gateway=None, session=None):
        cfg = Settings()

        if session is not None:
            self._nav_session = session
            self._owns_nav_session = False
        else:
            self._nav_session = open_zenoh_session(cfg.ZENOH_ROUTER)
            self._owns_nav_session = True
            logger.info("ExploreAgent: Zenoh nav session opened")

        self.pathplanner = pathplanner or PathPlannerAgent(session=self._nav_session)
        self.gateway = gateway or _build_gateway()

        self.route = []
        self.active = False
        self.blocked = False
        self.current_waypoint_index = 0
        self._field_size: dict | None = None
        # Cleared = paused, Set = free to move
        self._ready_to_move = threading.Event()
        self._ready_to_move.set()
        # Set to interrupt a running PointNavigator
        self._nav_stop = threading.Event()
        # Serialises BLOCK against the start of a navigation so a BLOCK that
        # arrives just before move_to_waypoint can't be wiped by its nav_stop.clear().
        self._block_lock = threading.Lock()

    def handle_request(self, request):
        request_type = request.get("type")

        if request_type == "START_EXPLORATION":
            return self.start_exploration(request)

        if request_type == "STOP_EXPLORATION":
            return self.stop_exploration()

        if request_type == "BLOCK":
            return self.block_exploration(request)

        if request_type == "UNBLOCK":
            return self.unblock_exploration()

        return {
            "status": "error",
            "agent": "explore_agent",
            "message": "Unknown request type"
        }

    def start_exploration(self, request):
        if "field_size" not in request:
            return {
                "status": "error",
                "agent": "explore_agent",
                "message": "Missing field_size"
            }

        self.active = True
        self.blocked = False
        self.current_waypoint_index = 0
        self._field_size = request["field_size"]

        return self.execute_frontier_loop()

    def move_to_waypoint(self, waypoint, timeout_s: float = WAYPOINT_TIMEOUT_S) -> bool:
        # Atomically check the block state and arm the navigator: if a BLOCK is
        # active (or races in here), don't start moving — return so the loop waits.
        with self._block_lock:
            if self.blocked:
                return False
            self._nav_stop.clear()
        nav = PointNavigator(
            target_x=waypoint["x"],
            target_y=waypoint["y"],
            gateway=self.gateway,
        )
        return nav.run(
            stop_event=self._nav_stop,
            timeout_s=timeout_s,
            zenoh_session=self._nav_session,
            collision_check=self._collision_check,
        )

    def _collision_check(self, x: float, y: float, yaw_deg: float) -> bool:
        """Forward-clearance probe used by the navigator to avoid driving into obstacles."""
        check = getattr(self.pathplanner, "is_forward_clear", None)
        if check is None:
            return True
        try:
            return check(x, y, yaw_deg)
        except Exception:
            logger.debug("ExploreAgent: collision check failed", exc_info=True)
            return True

    def stop_robot(self) -> None:
        self.gateway.publish_movement(MovementCommand(source=MovementSource.autonomous))

    def execute_frontier_loop(self):
        """Repeatedly fetch the next frontier and navigate to it until the field is covered."""
        while self.active:
            # Wait if movement is currently blocked
            self._ready_to_move.wait()

            if not self.active:
                break

            result = self.pathplanner.handle_request({
                "type": "NEXT_FRONTIER",
                "from": "explore_agent",
                "field_size": self._field_size,
            })

            if result.get("status") == "completed":
                logger.info("ExploreAgent: full area explored.")
                break

            if result.get("status") != "success":
                logger.warning("ExploreAgent: planner error — %s", result.get("message"))
                break

            self.route = result.get("waypoints", [])

            # Navigate through all A* waypoints leading to the frontier
            for wp_idx, waypoint in enumerate(self.route):
                if not self.active:
                    break

                self._ready_to_move.wait()
                if not self.active:
                    break

                reached = self.move_to_waypoint(waypoint)
                logger.info("ExploreAgent: waypoint %s reached=%s", waypoint["id"], reached)

                if not reached and self.active:
                    # Timeout or BLOCK — outer loop will re-plan with latest LiDAR data
                    break

                if not reached:
                    self.active = False
                    break

                self.current_waypoint_index += 1

                # After each step, verify remaining waypoints are still obstacle-free.
                # New LiDAR data may have revealed obstacles on the planned path.
                remaining = self.route[wp_idx + 1:]
                if remaining and not self.pathplanner.is_path_valid(remaining):
                    logger.info(
                        "ExploreAgent: remaining path blocked by new LiDAR data — re-planning"
                    )
                    break

        self.active = False

        return {
            "status": "completed",
            "agent": "explore_agent",
            "executed_waypoints": self.current_waypoint_index,
        }

    def block_exploration(self, request):
        with self._block_lock:
            self.blocked = True
            self._ready_to_move.clear()
            self._nav_stop.set()
        self.stop_robot()

        return {
            "status": "blocked",
            "agent": "explore_agent",
            "reason": request.get("reason", "unknown"),
            "current_waypoint_index": self.current_waypoint_index
        }

    def unblock_exploration(self):
        self.blocked = False
        self._ready_to_move.set()

        return {
            "status": "resumed",
            "agent": "explore_agent",
            "current_waypoint_index": self.current_waypoint_index
        }

    def stop_exploration(self):
        self.active = False
        self.blocked = False
        self._nav_stop.set()
        self._ready_to_move.set()
        self.stop_robot()
        if self._owns_nav_session:
            try:
                self._nav_session.close()
            except Exception:
                pass

        return {
            "status": "stopped",
            "agent": "explore_agent",
        }


if __name__ == "__main__":
    explore_agent = ExploreAgent()

    coordination_agent_request = {
        "type": "START_EXPLORATION",
        "from": "coordination_agent",
        "field_size": {
            "width_m": 5.0,
            "height_m": 5.0
        },
    }

    response = explore_agent.handle_request(coordination_agent_request)
    print(response)
