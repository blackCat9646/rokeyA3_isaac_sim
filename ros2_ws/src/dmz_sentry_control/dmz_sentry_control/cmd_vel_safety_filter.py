import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelSafetyFilter(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_safety_filter")

        self.declare_parameter("input_topic", "/cmd_vel_nav2_raw")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("max_linear_x", 0.65)
        self.declare_parameter("max_angular_z", 0.85)
        self.declare_parameter("angular_deadband", 0.04)

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._max_linear_x = max(0.05, float(self.get_parameter("max_linear_x").value))
        self._max_angular_z = max(0.05, float(self.get_parameter("max_angular_z").value))
        self._angular_deadband = max(0.0, float(self.get_parameter("angular_deadband").value))

        self._pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Twist, input_topic, self._on_cmd_vel, 10)
        self.get_logger().info(f"Filtering Nav2 cmd_vel: {input_topic} -> {output_topic}")

    def _on_cmd_vel(self, msg: Twist) -> None:
        filtered = Twist()
        angular = self._clamp(msg.angular.z, -self._max_angular_z, self._max_angular_z)

        if abs(angular) > self._angular_deadband:
            filtered.angular.z = angular
        else:
            filtered.linear.x = self._clamp(msg.linear.x, -self._max_linear_x, self._max_linear_x)

        if not math.isfinite(filtered.linear.x):
            filtered.linear.x = 0.0
        if not math.isfinite(filtered.angular.z):
            filtered.angular.z = 0.0

        self._pub.publish(filtered)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelSafetyFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
