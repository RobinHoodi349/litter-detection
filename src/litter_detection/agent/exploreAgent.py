from litter_detection.agent.pathPlanerAgent import PathPlannerAgent
from litter_detection.agent.move_agent import run_move_agent_sync
from litter_detection.agent.tools.motion_types import MoveAgentDeps


class ExploreAgent:
    def __init__(self, pathplanner=None, move_agent_deps=None):
        self.pathplanner = pathplanner or PathPlannerAgent()
        self.move_agent_deps = move_agent_deps or MoveAgentDeps.from_env()

        self.route = []
        self.active = False
        self.blocked = False
        self.current_waypoint_index = 0

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

        if "lane_spacing_m" not in request:
            return {
                "status": "error",
                "agent": "explore_agent",
                "message": "Missing lane_spacing_m"
            }

        self.active = True
        self.blocked = False
        self.current_waypoint_index = 0

        path_request = {
            "type": "PLAN_PATH",
            "from": "explore_agent",
            "field_size": request["field_size"],
            "lane_spacing_m": request["lane_spacing_m"]
        }

        path_response = self.pathplanner.handle_request(path_request)

        if path_response.get("status") != "success":
            self.active = False
            return {
                "status": "error",
                "agent": "explore_agent",
                "message": path_response.get("message", "Path planning failed")
            }

        self.route = path_response["waypoints"]

        print(
            f"ExploreAgent: received {len(self.route)} waypoints "
            f"from PathPlannerAgent"
        )

        return self.execute_route()

    def move_to_waypoint(self, waypoint):
        prompt = (
            "Fahre zur Zielkoordinate.\n"
            f"Waypoint-ID: {waypoint.get('id')}\n"
            f"x: {waypoint['x']}\n"
            f"y: {waypoint['y']}\n"
            "Quelle: explore_agent"
        )

        return run_move_agent_sync(prompt, deps=self.move_agent_deps)

    def stop_move_agent(self):
        prompt = "Stoppe sofort jede Bewegung. Quelle: explore_agent"
        return run_move_agent_sync(prompt, deps=self.move_agent_deps)

    def execute_route(self):
        while self.active and self.current_waypoint_index < len(self.route):
            if self.blocked:
                self.stop_move_agent()
                return {
                    "status": "blocked",
                    "agent": "explore_agent",
                    "current_waypoint_index": self.current_waypoint_index
                }

            waypoint = self.route[self.current_waypoint_index]

            move_result = self.move_to_waypoint(waypoint)

            print(f"ExploreAgent: MoveAgent result for {waypoint['id']}: {move_result}")

            if move_result is None or "error" in str(move_result).lower():
                self.active = False
                return {
                    "status": "error",
                    "agent": "explore_agent",
                    "message": f"MoveAgent failed at waypoint {waypoint['id']}",
                    "move_agent_result": str(move_result)
                }

            self.current_waypoint_index += 1

        self.active = False

        return {
            "status": "completed",
            "agent": "explore_agent",
            "executed_waypoints": self.current_waypoint_index
        }

    def block_exploration(self, request):
        self.blocked = True
        stop_result = self.stop_move_agent()

        return {
            "status": "blocked",
            "agent": "explore_agent",
            "reason": request.get("reason", "unknown"),
            "current_waypoint_index": self.current_waypoint_index,
            "move_agent_result": str(stop_result)
        }

    def unblock_exploration(self):
        self.blocked = False

        if self.active:
            return self.execute_route()

        return {
            "status": "idle",
            "agent": "explore_agent",
            "message": "No active exploration to resume"
        }

    def stop_exploration(self):
        self.active = False
        self.blocked = False
        stop_result = self.stop_move_agent()

        return {
            "status": "stopped",
            "agent": "explore_agent",
            "move_agent_result": str(stop_result)
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
        "lane_spacing_m": 0.5
    }

    response = explore_agent.handle_request(coordination_agent_request)
    print(response)