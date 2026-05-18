import json
import math
import time
from enum import Enum

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class PatrolMode(str, Enum):
    IDLE = "IDLE"
    PATROL = "PATROL"
    HOME = "HOME"
    ALERT_STOP = "ALERT_STOP"
    STOPPED = "STOPPED"


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class PatrolController(Node):
    def __init__(self) -> None:
        super().__init__("patrol_controller")

        self.declare_parameter("mission_topic", "/mission_command")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("alerts_topic", "/alerts")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("state_topic", "/patrol_state")
        self.declare_parameter("status_hz", 5.0)
        self.declare_parameter("patrol_y", -12.0)
        self.declare_parameter("patrol_x_min", -24.0)
        self.declare_parameter("patrol_x_max", 24.0)
        self.declare_parameter("home_x", 0.0)
        self.declare_parameter("home_y", 0.0)
        self.declare_parameter("waypoint_tolerance", 1.5)
        self.declare_parameter("max_linear_command", 1.0)
        self.declare_parameter("max_yaw_command", 1.0)
        self.declare_parameter("position_gain", 0.45)
        self.declare_parameter("yaw_gain", 1.6)
        self.declare_parameter("heading_tolerance", 0.35)
        self.declare_parameter("alert_hold_seconds", 6.0)

        self._mission_topic = self.get_parameter("mission_topic").get_parameter_value().string_value
        self._odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self._alerts_topic = self.get_parameter("alerts_topic").get_parameter_value().string_value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self._state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        status_hz = max(1.0, float(self.get_parameter("status_hz").value))

        self._patrol_y = float(self.get_parameter("patrol_y").value)
        self._patrol_x_min = float(self.get_parameter("patrol_x_min").value)
        self._patrol_x_max = float(self.get_parameter("patrol_x_max").value)
        self._home = (
            float(self.get_parameter("home_x").value),
            float(self.get_parameter("home_y").value),
        )
        self._waypoint_tolerance = max(0.1, float(self.get_parameter("waypoint_tolerance").value))
        self._max_linear_command = max(0.05, float(self.get_parameter("max_linear_command").value))
        self._max_yaw_command = max(0.05, float(self.get_parameter("max_yaw_command").value))
        self._position_gain = max(0.01, float(self.get_parameter("position_gain").value))
        self._yaw_gain = max(0.01, float(self.get_parameter("yaw_gain").value))
        self._heading_tolerance = max(0.02, float(self.get_parameter("heading_tolerance").value))
        self._alert_hold_seconds = max(0.0, float(self.get_parameter("alert_hold_seconds").value))

        self._mode = PatrolMode.IDLE
        self._resume_mode = PatrolMode.PATROL
        self._pose = None
        self._patrol_waypoints = (
            (self._patrol_x_min, self._patrol_y),
            (self._patrol_x_max, self._patrol_y),
        )
        self._lane_entry = (self._home[0], self._patrol_y)
        self._waypoint = self._home
        self._route_queue = []
        self._next_waypoint_index = 1
        self._last_alert_time = 0.0

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._state_pub = self.create_publisher(String, self._state_topic, 10)
        self.create_subscription(String, self._mission_topic, self._on_mission, 10)
        self.create_subscription(String, self._alerts_topic, self._on_alert, 10)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
        self.create_timer(1.0 / status_hz, self._tick)

        self.get_logger().info(
            "Patrol controller ready: "
            f"mission={self._mission_topic}, odom={self._odom_topic}, "
            f"cmd_vel={self._cmd_vel_topic}, state={self._state_topic}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._pose = (float(position.x), float(position.y), float(yaw))

    def _on_mission(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if command in ("start", "start_patrol", "launch", "sortie"):
            self._mode = PatrolMode.PATROL
            self._set_patrol_route_from_current_pose()
            self._resume_mode = PatrolMode.PATROL
            self.get_logger().info(
                f"Mission command: start tower patrol toward x={self._waypoint[0]:.1f}, y={self._waypoint[1]:.1f}"
            )
        elif command in ("home", "go_home", "return_home", "rtb"):
            self._mode = PatrolMode.HOME
            self._set_home_route_from_current_pose()
            self._resume_mode = PatrolMode.HOME
            self.get_logger().info(
                f"Mission command: return home toward x={self._waypoint[0]:.1f}, y={self._waypoint[1]:.1f}"
            )
        elif command in ("stop", "halt"):
            self._mode = PatrolMode.STOPPED
            self.get_logger().info("Mission command: stop")
        elif command in ("resume", "continue"):
            self._mode = self._resume_mode if self._pose is not None else PatrolMode.IDLE
            self.get_logger().info(f"Mission command: resume -> {self._mode.value}")
        elif command in ("idle", "standby"):
            self._mode = PatrolMode.IDLE
            self.get_logger().info("Mission command: idle")
        else:
            self.get_logger().warn(f"Unknown mission command: {msg.data}")

    def _on_alert(self, msg: String) -> None:
        self._last_alert_time = time.monotonic()
        if self._mode not in (PatrolMode.IDLE, PatrolMode.STOPPED, PatrolMode.ALERT_STOP):
            self._resume_mode = self._mode
        if self._mode != PatrolMode.STOPPED:
            self._mode = PatrolMode.ALERT_STOP
        try:
            payload = json.loads(msg.data)
            confidence = float(payload.get("confidence", 0.0))
            self.get_logger().warn(f"Alert received; holding patrol, confidence={confidence:.2f}")
        except Exception:
            self.get_logger().warn("Alert received; holding patrol")

    def _select_initial_patrol_waypoint(self) -> None:
        midpoint_x = 0.5 * (self._patrol_x_min + self._patrol_x_max)
        if self._pose is not None and self._pose[0] > midpoint_x:
            self._waypoint = self._patrol_waypoints[0]
            self._next_waypoint_index = 1
        else:
            self._waypoint = self._patrol_waypoints[1]
            self._next_waypoint_index = 0

    def _set_route(self, waypoints: list[tuple[float, float]]) -> None:
        if not waypoints:
            waypoints = [self._home]
        self._waypoint = waypoints[0]
        self._route_queue = list(waypoints[1:])

    def _set_patrol_route_from_current_pose(self) -> None:
        self._select_initial_patrol_waypoint()
        patrol_target = self._waypoint
        route = []
        if self._pose is None or abs(self._pose[1] - self._patrol_y) > self._waypoint_tolerance:
            route.append(self._lane_entry)
        route.append(patrol_target)
        self._set_route(route)

    def _set_home_route_from_current_pose(self) -> None:
        route = []
        if self._pose is None or abs(self._pose[0] - self._home[0]) > 3.0 or self._pose[1] > self._home[1] + 3.0:
            route.append(self._lane_entry)
        route.append(self._home)
        self._set_route(route)

    def _next_patrol_waypoint(self) -> tuple[float, float]:
        waypoint = self._patrol_waypoints[self._next_waypoint_index % len(self._patrol_waypoints)]
        self._next_waypoint_index += 1
        return waypoint

    def _advance_waypoint_after_arrival(self) -> None:
        if self._route_queue:
            self._waypoint = self._route_queue.pop(0)
            self.get_logger().info(
                f"Advancing route waypoint to x={self._waypoint[0]:.1f}, y={self._waypoint[1]:.1f}"
            )
            return

        if self._mode == PatrolMode.HOME:
            self._mode = PatrolMode.IDLE
            self._waypoint = self._home
            self.get_logger().info("Home reached; switching to IDLE")
            return

        self._waypoint = self._next_patrol_waypoint()
        self.get_logger().info(
            f"Switching patrol waypoint to x={self._waypoint[0]:.1f}, y={self._waypoint[1]:.1f}"
        )

    def _distance_to_waypoint(self) -> float | None:
        if self._pose is None:
            return None
        dx = self._waypoint[0] - self._pose[0]
        dy = self._waypoint[1] - self._pose[1]
        return math.hypot(dx, dy)

    def _make_tracking_command(self) -> Twist:
        msg = Twist()
        if self._pose is None:
            return msg

        x, y, yaw = self._pose
        dx_world = self._waypoint[0] - x
        dy_world = self._waypoint[1] - y
        distance = math.hypot(dx_world, dy_world)
        if distance < 1e-6:
            return msg

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dx_body = cos_yaw * dx_world + sin_yaw * dy_world

        desired_yaw = math.atan2(dy_world, dx_world)
        yaw_error = _wrap_angle(desired_yaw - yaw)
        if abs(yaw_error) > self._heading_tolerance:
            msg.angular.z = float(
                max(-self._max_yaw_command, min(self._max_yaw_command, yaw_error * self._yaw_gain))
            )
            return msg

        msg.linear.x = float(
            max(-self._max_linear_command, min(self._max_linear_command, dx_body * self._position_gain))
        )
        msg.linear.y = 0.0
        return msg

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _tick(self) -> None:
        now = time.monotonic()

        if self._mode == PatrolMode.ALERT_STOP:
            self._publish_stop()
            if now - self._last_alert_time > self._alert_hold_seconds:
                self._mode = self._resume_mode
                self.get_logger().info(f"Alert hold cleared; resuming {self._mode.value}")
        elif self._mode in (PatrolMode.IDLE, PatrolMode.STOPPED):
            self._publish_stop()
        else:
            distance = self._distance_to_waypoint()
            if distance is not None and distance <= self._waypoint_tolerance:
                self._advance_waypoint_after_arrival()
            if self._mode in (PatrolMode.IDLE, PatrolMode.STOPPED):
                self._publish_stop()
            else:
                self._cmd_pub.publish(self._make_tracking_command())

        self._publish_state()

    def _publish_state(self) -> None:
        payload = {
            "mode": self._mode.value,
            "waypoint": {"x": self._waypoint[0], "y": self._waypoint[1]},
            "home": {"x": self._home[0], "y": self._home[1]},
            "route": [{"x": waypoint[0], "y": waypoint[1]} for waypoint in self._route_queue],
            "pose": None,
        }
        if self._pose is not None:
            payload["pose"] = {"x": self._pose[0], "y": self._pose[1], "yaw": self._pose[2]}
        msg = String()
        msg.data = json.dumps(payload)
        self._state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PatrolController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
