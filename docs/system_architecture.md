# System Architecture

## Overview

The system is a multimodal human–robot interaction (HRI) framework designed to interpret and execute structured robot commands using voice and gesture input. It supports three interaction modes:

- Voice-only
- Gesture-only
- Multimodal (voice and gesture fusion)

The architecture is modular, allowing each modality to be processed independently before being combined through a fusion layer. The system is implemented as an interactive application using Streamlit, enabling both real-time interaction and structured experimental evaluation.

The system is designed to integrate with assistive robotic platforms such as the [Hello Robot](https://hello-robot.com/) Stretch 3 robot, supporting intuitive command input for pick-and-place tasks.

## High-Level Pipeline

The system follows a sequential processing pipeline:

1. **Input Acquisition**
   - Voice input (microphone or typed text)
   - Gesture input (webcam or button-based fallback)

2. **Modality-Specific Processing**
   - Voice parsing to structured command
   - Gesture detection to action or location inference

3. **Multimodal Fusion**
   - Combines voice and gesture outputs
   - Resolves conflicts and missing fields
   - Applies temporal and confidence-based reasoning
   - *(See [`fusion_design.md`](fusion_design.md) for full details)*

4. **Command Output**
   - Final structured command (action, object, location)
   - Confidence score and metadata

5. **Robot Dispatch (Live Mode Only)**
   - Fused command dispatched over WiFi to the Stretch 3 robot
   - Bridge publishes command as a ROS 2 message
   - Executor maps command to a hardcoded motion primitive
   - *(See [`ros2_integration.md`](ros2_integration.md) for full details)*

6. **Evaluation Layer (Trial Mode Only)**
   - Compares predicted vs expected command
   - Logs results for analysis

## System Architecture Diagram

The system architecture consists of parallel voice and gesture processing pipelines, whose outputs are combined using decision-level fusion to produce a structured command. A Streamlit-based interaction layer manages both live interaction and experiment modes, while experimental trials are logged for offline analysis.

![System Architecture Diagram](/assets/github-system-architecture.png)


## Component Architecture

### Voice Module

The voice module converts natural language input into structured robot commands.

**Responsibilities:**
- Transcribe speech (Whisper model)
- Parse text into structured intent (`action`, `object`, `location`)
- Assign confidence scores per field
- Validate command completeness

**Key Files:**
- `voice/parser.py`
- `voice/speech.py`
- `voice/validation.py`

---
### Gesture Module

The gesture module interprets hand gestures and spatial position from webcam input.

**Responsibilities:**
- Detect hand landmarks using MediaPipe
- Classify gesture type (e.g., pick, stop, place)
- Infer spatial location (left/right)
- Track gesture stability over time

**Key Files:**
- `gesture/detector.py`
- `gesture/mapper.py`

---
### Fusion Module

The fusion module combines voice and gesture inputs into a single command.

**Responsibilities:**
- Align inputs within a temporal window
- Resolve conflicts between modalities
- Fill missing fields (e.g., location from gesture)
- Compute final confidence score
- Flag ambiguity and conflicts

**Key File:**
- `fusion/fuser.py`

*Full design and decision logic is described in [`fusion_design.md`](fusion_design.md)*

---
### Interaction Layer (User Interface)

The system uses Streamlit to provide two interaction modes:

#### Live Interaction Mode
- Real-time command input
- No logging or evaluation
- Supports all modalities

#### Trial / Experiment Mode
- Structured tasks with prompts
- Step-by-step interaction flow
- Accuracy, latency, and correction tracking
- Result logging

**Key Files:**
- `ui/components.py`
- `ui/streamlit_app.py`

---
## Interaction Flow Diagram
The following sequence diagram illustrates how the system processes user input in live interaction mode, including optional voice and gesture inputs and their fusion into a structured command.

![Live Mode Sequence Diagram](/assets/live-sequence-diagram.png)

### Experiment Management

Handles structured evaluation sessions.

**Responsibilities:**
- Load predefined trials
- Apply counterbalancing across participants
- Manage trial progression
- Compute correctness of responses

**Key Files:**
- `experiments/trials.py`
- `experiments/runner.py`

### Logging and Data Storage

All trial results are saved as CSV files for later analysis.

**Captured Data Includes:**
- Predicted vs expected commands
- Accuracy
- Latency
- Correction count
- Fusion metadata (conflict, timestamps, temporal window)
- Confidence scores

**Key File:**
- `trial_logger/logger.py`

### Analysis Module

Processes logged data to compute evaluation metrics and generate visualisations.

**Responsibilities:**
- Load and validate CSV session logs
- Compute metrics (accuracy, latency, error rates, etc.)
- Generate plots for evaluation

**Key Files:**
- `analysis/loader.py`
- `analysis/metrics.py`
- `analysis/plots.py`

---
## Data Flow
The data flow diagram illustrates how information moves through the system, from multimodal input to command generation, evaluation, and analysis.

![Data Flow Diagram](/assets/data-flow-diagram.png)

## Extensibility

The system is designed to support future extensions, including:

- Integration with additional physical robotic platforms beyond Stretch 3
- Dynamic motion planning on the Stretch 3 beyond hardcoded pose primitives
- Additional gesture vocabularies
- More complex multi-step commands
- Adaptive or learning-based fusion strategies

---