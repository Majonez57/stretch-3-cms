"""
Stretch-side ROS 2 bridge.

Runs on the Stretch robot to receive HTTP POSTs from the laptop and publish them as ROS 2 String messages on the /hri/command topic.

Run on Stretch (with ROS 2 Humble sourced):
    python3 ros2_bridge.py

Manual test from laptop:
    curl -s -X POST http://STRETCH_IP:5050/command \\
         -H "Content-Type: application/json" \\
         -d '{"action": "stop"}'

    curl -s http://STRETCH_IP:5050/status

Verify on Stretch:
    ros2 topic echo /hri/command
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

_VALID_ACTIONS = {"pick", "place", "move", "stop", "cancel"}
_HOST = "0.0.0.0" 
_PORT = 5050

# Global reference to the ROS node for use in the HTTP request handler.
_node: Optional[HRIBridgeNode] = None


class HRIBridgeNode(Node):
    """
    ROS 2 node that publishes commands to /hri/command.

    Attributes:
        _pub: ROS 2 publisher for String messages on /hri/command.
    """

    def __init__(self) -> None:
        """
        Initialise the ROS 2 node and publisher.
        """

        super().__init__("hri_bridge")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(String, "/hri/command", qos)

        self.get_logger().info("HRI bridge node started — publishing on /hri/command")

    def publish_command(self, payload: dict) -> None:
        """
        Publish a command payload as a JSON string message.

        Args:
            payload: Validated command dict.
        """

        msg = String()
        msg.data = json.dumps(payload)
        self._pub.publish(msg)
        
        self.get_logger().info(f"Published: {msg.data}")


class _CommandHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the bridge.

    Attributes:
        _node: Reference to the HRIBridgeNode for publishing commands.

    Endpoints:  
        POST /command — accepts and publishes a robot command
        GET /status — returns bridge health
    """

    def do_POST(self) -> None:
        """
        Handle incoming command POST requests.
        """

        if self.path != "/command":
            self._respond(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON"})
            return

        action = payload.get("action")
        if action not in _VALID_ACTIONS:
            self._respond(
                400,
                {
                    "error": f"invalid or missing 'action'; "
                    f"must be one of {sorted(_VALID_ACTIONS)}"
                },
            )
            return

        if _node is None:
            self._respond(503, {"error": "ROS node not ready"})
            return

        _node.publish_command(payload)
        self._respond(200, {"status": "accepted"})

    def do_GET(self) -> None:
        """
        Handle incoming status GET requests.
        """

        if self.path != "/status":
            self._respond(404, {"error": "not found"})
            return
        self._respond(
            200,
            {
                "status": "ok",
                "node": "hri_bridge",
                "topic": "/hri/command",
            },
        )

    def _respond(self, code: int, body: dict) -> None:
        """
        Send a JSON response with the given HTTP status code and body.
        """

        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        """
        Log HTTP requests using the node's logger instead of printing to stderr.
        """

        print(f"[bridge] {self.address_string()} — {fmt % args}")


def _spin_ros(node: Node) -> None:
    """
    Spin the ROS 2 node in a separate thread to handle incoming commands.

    Args:
        node: The HRIBridgeNode instance to spin.
    """
    rclpy.spin(node)


def main() -> None:
    """
    Main entry point for the ROS 2 bridge.

    Initializes the ROS 2 node, starts the HTTP server, and handles shutdown.
    """

    global _node

    rclpy.init()
    _node = HRIBridgeNode()

    ros_thread = threading.Thread(target=_spin_ros, args=(_node,), daemon=True)
    ros_thread.start()

    server = HTTPServer((_HOST, _PORT), _CommandHandler)
    print(f"[bridge] HTTP server listening on 0.0.0.0:{_PORT} (all interfaces)")
    print(f"[bridge] Publishing to /hri/command — waiting for commands…")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] Shutting down")
    finally:
        server.server_close()
        _node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
