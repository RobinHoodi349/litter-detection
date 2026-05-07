import threading

from litter_detection.agent.pathPlanerAgent import PathPlannerAgent
from litter_detection.agent.navigator import PointNavigator
from litter_detection.agent.models import MovementCommand, MovementSource
from litter_detection.agent.tools.motion_types import RobotMotionGateway
from litter_detection.config import Settings


def _build_gateway() -> RobotMotionGateway:
    cfg = Settings()
    return RobotMotionGateway(
        router=cfg.ZENOH_ROUTER,
        movement_topic=cfg.topic_movement_command,
        dry_run=False,
    )


class ExploreAgent:
    def __init__(self, pathplanner=None, gateway=None):
        self.pathplanner = pathplanner or PathPlannerAgent()
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

    def move_to_waypoint(self, waypoint) -> bool:
        self._nav_stop.clear()
        nav = PointNavigator(
            target_x=waypoint["x"],
            target_y=waypoint["y"],
            gateway=self.gateway,
        )
        return nav.run(stop_event=self._nav_stop)

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
                print("ExploreAgent: full area explored.")
                break

            if result.get("status") != "success":
                print(f"ExploreAgent: planner error — {result.get('message')}")
                break

            self.route = result.get("waypoints", [])

            # Navigate through all A* waypoints leading to the frontier
            for waypoint in self.route:
                if not self.active:
                    break

                self._ready_to_move.wait()
                if not self.active:
                    break

                reached = self.move_to_waypoint(waypoint)
                print(f"ExploreAgent: waypoint {waypoint['id']} reached={reached}")

                if not reached and self.active:
                    # Interrupted mid-route (BLOCK); outer loop will re-plan
                    break

                if not reached:
                    self.active = False
                    break

                self.current_waypoint_index += 1

        self.active = False

        return {
            "status": "completed",
            "agent": "explore_agent",
            "executed_waypoints": self.current_waypoint_index,
        }

    def block_exploration(self, request):
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
