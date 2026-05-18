import json
import math
import time
from enum import Enum

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


class MissionMode(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_NAV2 = "WAITING_FOR_NAV2"
    PATROL = "PATROL"
    HOME = "HOME"
    ALERT_STOP = "ALERT_STOP"
    STOPPED = "STOPPED"


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class Nav2PatrolController(Node):
    def __init__(self) -> None:
        super().__init__("nav2_patrol_controller")

        self.declare_parameter("mission_topic", "/mission_command")
        self.declare_parameter("alerts_topic", "/alerts")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("state_topic", "/patrol_state")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("action_name", "navigate_to_pose")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("home_x", 0.0)
        self.declare_parameter("home_y", 0.0)
        self.declare_parameter("lane_entry_x", 0.0)
        self.declare_parameter("lane_entry_y", -12.0)
        self.declare_parameter("patrol_x_min", -24.0)
        self.declare_parameter("patrol_x_max", 24.0)
        self.declare_parameter("patrol_y", -12.0)
        self.declare_parameter("status_hz", 5.0)
        self.declare_parameter("alert_hold_seconds", 6.0)

        self._mission_topic = self.get_parameter("mission_topic").get_parameter_value().string_value
        self._alerts_topic = self.get_parameter("alerts_topic").get_parameter_value().string_value
        self._odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self._state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self._action_name = self.get_parameter("action_name").get_parameter_value().string_value
        self._global_frame = self.get_parameter("global_frame").get_parameter_value().string_value
        self._home = (
            float(self.get_parameter("home_x").value),
            float(self.get_parameter("home_y").value),
        )
        self._lane_entry = (
            float(self.get_parameter("lane_entry_x").value),
            float(self.get_parameter("lane_entry_y").value),
        )
        self._patrol_y = float(self.get_parameter("patrol_y").value)
        self._patrol_waypoints = (
            (float(self.get_parameter("patrol_x_min").value), self._patrol_y),
            (float(self.get_parameter("patrol_x_max").value), self._patrol_y),
        )
        status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self._alert_hold_seconds = max(0.0, float(self.get_parameter("alert_hold_seconds").value))

        self._mode = MissionMode.IDLE
        self._resume_mode = MissionMode.PATROL
        self._pose = None
        self._current_goal = None
        self._route_queue = []
        self._next_patrol_index = 1
        self._goal_handle = None
        self._last_alert_time = 0.0

        self._nav_client = ActionClient(self, NavigateToPose, self._action_name)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._state_pub = self.create_publisher(String, self._state_topic, 10)
        self.create_subscription(String, self._mission_topic, self._on_mission, 10)
        self.create_subscription(String, self._alerts_topic, self._on_alert, 10)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
        self.create_timer(1.0 / status_hz, self._tick)

        self.get_logger().info(
            "Nav2 patrol controller ready: "
            f"mission={self._mission_topic}, action={self._action_name}, frame={self._global_frame}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._pose = (float(position.x), float(position.y), float(yaw))

    def _on_mission(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if command in ("start", "start_patrol", "launch", "sortie"):
            self._mode = MissionMode.PATROL
            self._resume_mode = MissionMode.PATROL
            self._set_patrol_route()
            self._send_next_goal()
        elif command in ("home", "go_home", "return_home", "rtb"):
            self._mode = MissionMode.HOME
            self._resume_mode = MissionMode.HOME
            self._set_home_route()
            self._send_next_goal()
        elif command in ("stop", "halt"):
            self._mode = MissionMode.STOPPED
            self._route_queue = []
            self._cancel_current_goal()
            self._publish_stop()
            self.get_logger().info("Mission command: stop")
        elif command in ("resume", "continue"):
            if self._mode in (MissionMode.ALERT_STOP, MissionMode.STOPPED):
                self._mode = self._resume_mode
                if not self._current_goal:
                    self._set_patrol_route() if self._mode == MissionMode.PATROL else self._set_home_route()
                self._send_next_goal()
            self.get_logger().info(f"Mission command: resume -> {self._mode.value}")
        elif command in ("idle", "standby"):
            self._mode = MissionMode.IDLE
            self._route_queue = []
            self._cancel_current_goal()
            self._publish_stop()
            self.get_logger().info("Mission command: idle")
        else:
            self.get_logger().warn(f"Unknown mission command: {msg.data}")

    def _on_alert(self, msg: String) -> None:
        self._last_alert_time = time.monotonic()
        if self._mode not in (MissionMode.IDLE, MissionMode.STOPPED, MissionMode.ALERT_STOP):
            self._resume_mode = self._mode
        if self._mode != MissionMode.STOPPED:
            self._mode = MissionMode.ALERT_STOP
            self._cancel_current_goal()
            self._publish_stop()
        try:
            payload = json.loads(msg.data)
            confidence = float(payload.get("confidence", 0.0))
            self.get_logger().warn(f"Alert received; Nav2 patrol holding, confidence={confidence:.2f}")
        except Exception:
            self.get_logger().warn("Alert received; Nav2 patrol holding")

    def _set_patrol_route(self) -> None:
        target = self._select_initial_patrol_target()
        route = []
        if self._pose is None or abs(self._pose[1] - self._patrol_y) > 1.5:
            route.append(self._lane_entry)
        route.append(target)
        self._route_queue = route
        self._current_goal = None

    def _set_home_route(self) -> None:
        route = []
        if self._pose is None or abs(self._pose[0] - self._home[0]) > 3.0 or self._pose[1] > self._home[1] + 3.0:
            route.append(self._lane_entry)
        route.append(self._home)
        self._route_queue = route
        self._current_goal = None

    def _select_initial_patrol_target(self) -> tuple[float, float]:
        midpoint_x = 0.5 * (self._patrol_waypoints[0][0] + self._patrol_waypoints[1][0])
        if self._pose is not None and self._pose[0] > midpoint_x:
            self._next_patrol_index = 1
            return self._patrol_waypoints[0]
        self._next_patrol_index = 0
        return self._patrol_waypoints[1]

    def _next_patrol_target(self) -> tuple[float, float]:
        waypoint = self._patrol_waypoints[self._next_patrol_index % len(self._patrol_waypoints)]
        self._next_patrol_index += 1
        return waypoint

    def _send_next_goal(self) -> None:
        if self._mode in (MissionMode.IDLE, MissionMode.STOPPED, MissionMode.ALERT_STOP):
            return

        if self._goal_handle is not None:
            return

        if not self._nav_client.server_is_ready():
            self._mode = MissionMode.WAITING_FOR_NAV2
            self.get_logger().warn("Nav2 navigate_to_pose action is not ready yet")
            return

        if not self._route_queue:
            if self._mode == MissionMode.HOME:
                self._mode = MissionMode.IDLE
                self._current_goal = self._home
                self._publish_stop()
                self.get_logger().info("Home reached; switching to IDLE")
                return
            self._route_queue.append(self._next_patrol_target())

        self._current_goal = self._route_queue.pop(0)
        goal = NavigateToPose.Goal()
        goal.pose = self._make_pose(self._current_goal)
        self.get_logger().info(
            f"Sending Nav2 goal x={self._current_goal[0]:.1f}, y={self._current_goal[1]:.1f}"
        )
        send_future = self._nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def _make_pose(self, point: tuple[float, float]) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self._global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.orientation = _quaternion_from_yaw(0.0)
        return pose

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            self._goal_handle = None
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        self._goal_handle = None
        status = future.result().status
        if self._mode in (MissionMode.ALERT_STOP, MissionMode.STOPPED, MissionMode.IDLE):
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Nav2 goal reached")
            self._send_next_goal()
        else:
            self.get_logger().warn(f"Nav2 goal ended with status={status}; trying next cycle")
            self._send_next_goal()

    def _cancel_current_goal(self) -> None:
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _tick(self) -> None:
        now = time.monotonic()
        if self._mode == MissionMode.WAITING_FOR_NAV2 and self._nav_client.server_is_ready():
            self._mode = self._resume_mode
            self._send_next_goal()
        elif self._mode == MissionMode.ALERT_STOP:
            self._publish_stop()
            if now - self._last_alert_time > self._alert_hold_seconds:
                self._mode = self._resume_mode
                self.get_logger().info(f"Alert hold cleared; resuming {self._mode.value}")
                self._send_next_goal()

        self._publish_state()

    def _publish_state(self) -> None:
        payload = {
            "mode": self._mode.value,
            "waypoint": {"x": self._current_goal[0], "y": self._current_goal[1]} if self._current_goal else None,
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
    node = Nav2PatrolController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
