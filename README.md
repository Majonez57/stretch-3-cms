# Stretch-3 CMS Demo: r-WALTER integration

Repo containing the code and instructions to run the Stretch-3 (Mr. Fantastic) pick-and-place demo. 
This demo can also be ran using the r-WALTER biocabinet.

The demo aims to allow a user to select any given object in the robot view (either using a click, or hand-tracking), and the robot will use [visual servoing](https://github.com/hello-robot/stretch_visual_servoing/tree/main) to grasp the object and pick it up.

The demo uses:
    - MobileSAM to segement arbitrary objects
    - Norfair to track objects in the robot camera view
    - Visual servoing to approach wanted object, and grasp it

## Installation

- Install the `hand_landmarker.task` file from [mediapipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) and place it in the`/gestures/` folder

## Running the Demo

**Physical checks and other requirements**

- Make sure that the IP of your remote machine is on line 2 of `stretch_visual_servoing/yolo_networking.py` (on the robot)

**Running the demo:**
- To begin publishing images, on the robot terminal A:
```bash
cd stretch-3-cms/robot/
python3 send_camera_images.py
``` 
- To engage the visual servoing, in robot terminal B:
> [!WARNING]
> When you run this command, the robot will move into the initial position for grasping. Always check your surroundings
```bash
cd Documents/stretch_visual_servoing/
python3 visual_servoing_demo.py -y -r
```
> [!IMPORTANT]
> In the case of a failure, or emergency where the demo needs to be immediatly stopped, kill this terminal!

- To run the demo, on the Jetson (or another GPU device):
```bash
cd Documents/stretch-3-cms/
python3 -m vision.run_demo
```

You should see a GUI with two images. To use gesture selection, face the webcam, and point with your finger. You will see a cursor appear on the robot view. Once you are happy with the selected object, lift your middle finger to select.

Alternatively, simply click on the object you want to grasp in the robot view.
<!--
## Project Structure

The project is organised into modular components:

```bash
├── app.py                          # Main application entry point
├── config.py                       # Global config constants
├── models.py                       # Data models
│ 
├── voice                           # Voice processing module
│   ├── parser.py                   # NL to structured command
│   ├── speech.py                   # Speech-to-text (Whisper)
│   └── validation.py               # Command validation
│
├── gesture                         # Gesture processing module
│   ├── detector.py                 # Hand tracking and gesture detection
│   ├── keyboard_fallback.py        # Keyboard fallback
│   ├── mapper.py                   # Maps keyboard inputs to commands
│   └── sequence.py                 # Manages input flow
│
├── fusion                          # Multimodal fusion module
│   └── fuser.py                
│
├── ui                              # Streamlit UI
│   ├── components.py               # Reusable UI elements
│   └── streamlit_app.py            # Streamlit UI, managing live interaction and experiment modes
│
├── experiments                     # Experiment execution logic
│   ├── runner.py
│   ├── trial_definitions.json
│   └── trials.py
│
├── trial_logger                    # Logging system
│   └── logger.py
│ 
├── analysis                        # Data analysis pipeline
│   ├── loader.py
│   ├── metrics.py
│   ├── plots.py
│   ├── run_analysis.py
│   └── stats.py
│
├── tests                           # Unit tests
│   ├── test_analysis_loader.py
│   ├── test_analysis_metrics.py
│   └── ...
│
├── docs                            # Documentation
│   ├── analysis.md
│   ├── experiment_design.md
│   ├── fusion_design.md
│   ├── ros2_integration.md
│   └── system_architecture.md
│ 
├── assets                          # Images
│
└── logs                            # Logs (generated at runtime)
```
-->

## Notes

- Webcam access is required for gesture input.

## Authors

- Adrian Vecina Tercero 
- Ruben Odamo

## Aknowledgements

- Hello Robot for the visual servoing scripts

AI was used in parts of this project.