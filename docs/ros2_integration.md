# ROS 2 Integration

## Overview

The system integrates with the Hello Robot Stretch 3 robot via a HTTP bridge running on the robot. The Streamlit app on the laptop dispatches fused commands over WiFi; the bridge publishes them as ROS 2 messages on the robot, where a subscriber maps them to hardcoded motion primitives.

![ROS 2 Integration Diagram](/assets/ros2-integration-diagram.png)

## Running

### Prerequisites

1. Update `config.py` on the laptop:
   ```python
   ROS_DISPATCH_ENABLED = True
   ROS_BRIDGE_URL = "http://<STRETCH_IP>:5050"
   ```

2. Copy files to the Stretch:
   ```bash
   scp ros2/ros2_bridge.py ros2/stretch_executor.py hello-robot@<STRETCH_IP>:~/multimodal-hri/
   ```

### On Stretch
**Session 1 (bridge):**
```bash
source /opt/ros/humble/setup.bash
cd ~/multimodal-hri
python3 ros2_bridge.py
```

**Session 2 (executor):**
```bash
source /opt/ros/humble/setup.bash
cd ~/multimodal-hri
python3 stretch_executor.py
```

**On laptop:**
```bash
streamlit run app.py
```

## Physical Setup

- Table positioned directly in front of the robot, approximately 40 cm from the arm in homed position
- Two object positions marked on the table: **L** (left) and **R** (right)
- Left/right displacement achieved via base rotation combined with fixed arm extension

![Physical Setup Image](/assets/setup.jpeg)

---