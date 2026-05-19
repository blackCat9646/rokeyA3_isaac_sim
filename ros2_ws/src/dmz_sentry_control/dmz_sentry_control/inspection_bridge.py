import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


COMMAND_FILE = Path("/tmp/dmz_sentry_inspection_command.json")
INTRUDER_STATES_FILE = Path("/tmp/dmz_sentry_intruder_states.json")


class InspectionBridge(Node):
    def __init__(self) -> None:
        super().__init__("dmz_sentry_inspection_bridge")

        self.declare_parameter("command_topic", "/inspection_camera/command")
        self.declare_parameter("intruder_states_topic", "/intruder_states")
        self.declare_parameter("poll_hz", 10.0)

        command_topic = self.get_parameter("command_topic").get_parameter_value().string_value
        intruder_states_topic = self.get_parameter("intruder_states_topic").get_parameter_value().string_value
        poll_hz = max(1.0, float(self.get_parameter("poll_hz").value))

        self._sequence = 0
        self._last_intruder_payload = None
        self._last_intruder_publish_time = 0.0

        self._intruder_pub = self.create_publisher(String, intruder_states_topic, 10)
        self.create_subscription(String, command_topic, self._on_command, 10)
        self.create_timer(1.0 / poll_hz, self._tick)

        self.get_logger().info(
            "Inspection bridge ready: "
            f"{command_topic} -> {COMMAND_FILE}, {INTRUDER_STATES_FILE} -> {intruder_states_topic}"
        )

    def _on_command(self, msg: String) -> None:
        self._sequence += 1
        payload = {
            "sequence": self._sequence,
            "stamp": time.time(),
            "data": msg.data,
        }
        try:
            COMMAND_FILE.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to write inspection command file: {exc}")

    def _tick(self) -> None:
        try:
            payload = INTRUDER_STATES_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except Exception as exc:
            self.get_logger().warn(f"Failed to read intruder state file: {exc}")
            return

        now = time.monotonic()
        if payload == self._last_intruder_payload and now - self._last_intruder_publish_time < 0.5:
            return

        msg = String()
        msg.data = payload
        self._intruder_pub.publish(msg)
        self._last_intruder_payload = payload
        self._last_intruder_publish_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InspectionBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
