import json
import time
import zenoh


class ZenohRobotLocalizationClient:
    def __init__(self, topic="robot/state/pose", timeout_s=2.0):
        self.topic = topic
        self.timeout_s = timeout_s
        self.latest_pose = None

        self.session = zenoh.open(zenoh.Config())
        self.subscriber = self.session.declare_subscriber(
            self.topic,
            self._on_pose
        )

    def _on_pose(self, sample):
        payload = sample.payload.to_bytes().decode("utf-8")
        data = json.loads(payload)

        self.latest_pose = {
            "x": float(data["x"]),
            "y": float(data["y"]),
            "theta_deg": float(data.get("theta_deg", 0.0))
        }

    def get_current_pose(self):
        start_time = time.time()

        while self.latest_pose is None:
            if time.time() - start_time > self.timeout_s:
                raise TimeoutError(
                    f"No robot pose received on Zenoh topic '{self.topic}'"
                )
            time.sleep(0.05)

        return self.latest_pose


class PathPlannerAgent:
    def __init__(self, localization_client=None):
        self.localization_client = localization_client or ZenohRobotLocalizationClient()

    def validate_parameters(self, width, height, lane_spacing):
        if width <= 0 or height <= 0 or lane_spacing <= 0:
            raise ValueError("width, height and lane_spacing must be positive values")

        if lane_spacing > height:
            raise ValueError("lane_spacing must not be larger than height")

    def handle_request(self, request):
        if request.get("type") != "PLAN_PATH":
            return {
                "status": "error",
                "agent": "pathplanner",
                "message": "Unknown request type"
            }

        if "field_size" not in request:
            return {
                "status": "error",
                "agent": "pathplanner",
                "message": "Missing field_size"
            }

        if "lane_spacing_m" not in request:
            return {
                "status": "error",
                "agent": "pathplanner",
                "message": "Missing lane_spacing_m"
            }

        field_size = request["field_size"]
        width = field_size["width_m"]
        height = field_size["height_m"]
        lane_spacing = request["lane_spacing_m"]

        self.validate_parameters(width, height, lane_spacing)

        try:
            robot_pose = self.localization_client.get_current_pose()
        except Exception as error:
            return {
                "status": "error",
                "agent": "pathplanner",
                "message": f"Could not get robot pose: {error}"
            }

        return self.plan(
            start_x=robot_pose["x"],
            start_y=robot_pose["y"],
            theta_deg=robot_pose["theta_deg"],
            width=width,
            height=height,
            lane_spacing=lane_spacing
        )

    def plan(self, start_x, start_y, theta_deg, width, height, lane_spacing):
        waypoints = []
        direction = 1
        waypoint_id = 0

        number_of_lanes = int(height / lane_spacing)

        y_values = [
            round(i * lane_spacing, 3)
            for i in range(number_of_lanes + 1)
        ]

        if y_values[-1] < height:
            y_values.append(height)

        for y_offset in y_values:
            if direction == 1:
                row_points = [
                    (start_x, start_y + y_offset),
                    (start_x + width, start_y + y_offset)
                ]
            else:
                row_points = [
                    (start_x + width, start_y + y_offset),
                    (start_x, start_y + y_offset)
                ]

            for x, y in row_points:
                waypoints.append({
                    "id": f"wp_{waypoint_id}",
                    "x": round(x, 3),
                    "y": round(y, 3)
                })
                waypoint_id += 1

            direction *= -1

        return {
            "status": "success",
            "agent": "pathplanner",
            "target_agent": "explore_agent",
            "plan_type": "coverage_path",
            "pattern": "boustrophedon_lawnmower",
            "coordinate_frame": "world_absolute",
            "robot_start_pose": {
                "x": start_x,
                "y": start_y,
                "theta_deg": theta_deg
            },
            "search_zone": {
                "x_min": start_x,
                "x_max": start_x + width,
                "y_min": start_y,
                "y_max": start_y + height
            },
            "field_size": {
                "width_m": width,
                "height_m": height
            },
            "lane_spacing_m": lane_spacing,
            "safety": {
                "boundary_enforced": True,
                "collision_avoidance_required": False,
                "obstacle_free_field_assumed": True,
                "movement_commands_generated": False
            },
            "waypoints": waypoints
        }