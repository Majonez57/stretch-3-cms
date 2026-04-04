"""
Stretch 3 ROS 2 command executor — stub version (Phase 9, Step 4).

Subscribes to /hri/command, parses the received JSON payload, and logs the
command to stdout. No motion is performed in this stub version.

Use this to verify end-to-end connectivity (laptop → bridge → topic → Stretch)
before implementing motion primitives in Step 6.

Run on the Stretch robot (with ROS 2 Humble sourced):
    python3 stretch_executor.py

Expected output on receiving a command:
    [executor] Received command | action=pick object=red_cube location=left confidence=0.85
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class StretchExecutorNode(Node):
    """
    ROS 2 node that receives HRI commands and maps them to robot actions.

    Attributes:
        _pub: ROS 2 publisher for String messages on /hri/command.
    """

    def __init__(self) -> None:
        """
        Initialise the ROS 2 node and subscriber.
        """

        super().__init__("stretch_executor")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(String, "/hri/command", self._on_command, qos)

        self.get_logger().info(
            "Stretch executor ready — subscribed to /hri/command (stub mode)"
        )

    def _on_command(self, msg: String) -> None:
        """
        Handle an incoming command message.

        Args:
            msg: ROS 2 String message containing a JSON command payload.
        """
        
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON received: {msg.data!r}")
            return

        action = payload.get("action", "unknown")
        obj = payload.get("object") or "-"
        location = payload.get("location") or "—"
        confidence = float(payload.get("confidence", 0.0))

        self.get_logger().info(
            f"Received command | action={action} object={obj} "
            f"location={location} confidence={confidence:.2f}"
        )

        # TODO: Implement motion primitives, stub just logs the command for now.
        # Example structure:
        #
        # COMMAND_PRIMITIVES = {
        #     "stop":   self._stop_all,
        #     "cancel": self._stop_all,
        #     "pick":   self._pick_fixed,
        #     "place":  self._place_fixed,
        #     "move":   self._move_to_waypoint,
        # }
        # primitive = COMMAND_PRIMITIVES.get(action)
        # if primitive:
        #     primitive(payload)
        # else:
        #     self.get_logger().warn(f"No primitive for action: {action}")


def main() -> None:
    """
    Main entry point for the Stretch executor node.
    """

    rclpy.init()
    node = StretchExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
