"""
Command dispatcher for the ROS 2 bridge.

Sends RobotCommand instances to the ROS 2 bridge via HTTP.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

import config
from models import RobotCommand


def send_command(cmd: RobotCommand) -> bool:
    """
    POST a RobotCommand to the local ROS 2 bridge as JSON.

    Args:
        cmd: The confirmed robot command to dispatch.

    Returns:
        True if the bridge accepted the command (HTTP 200), False otherwise.
    """
    payload = {
        "action": cmd.action.value if cmd.action else None,
        "object": cmd.object.value if cmd.object else None,
        "location": cmd.location.value if cmd.location else None,
        "confidence": round(cmd.confidence, 4),
        "mode": cmd.mode.value,
        "timestamp": cmd.timestamp,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{config.ROS_BRIDGE_URL}/command"

    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as exc:
        print(f"[dispatch] Bridge unreachable at {url}: {exc}")
        return False


def dispatch_if_enabled(cmd: RobotCommand) -> Optional[bool]:
    """
    Dispatch to the bridge only if config.ROS_DISPATCH_ENABLED is True.

    Args:
        cmd: The confirmed robot command.

    Returns:
        True/False from send_command, or None if dispatch is disabled.
    """
    if not config.ROS_DISPATCH_ENABLED:
        return None
    return send_command(cmd)
