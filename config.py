"""
Global configuration constants for multimodal HRI system.

Fusion parameters:
    - FUSION_WINDOW_SECONDS: Time window (seconds) in which voice and gesture inputs are considered simultaneous for fusion.
    - FUSION_VOICE_CONFIDENCE_THRESHOLD: Minimum confidence required to accept voice-derived fields.
    - FUSION_GESTURE_ACTION_CONFIDENCE_THRESHOLD: Minimum confidence for gesture action recognition.
    - FUSION_GESTURE_LOCATION_CONFIDENCE_THRESHOLD: Minimum confidence for gesture-based location inference.
    - FUSION_VOICE_BIAS_MARGIN: Margin by which voice confidence must exceed gesture confidence to override it.
    - FUSION_AGREEMENT_BONUS: Confidence boost when voice and gesture agree.
    - FUSION_CONFLICT_PENALTY: Confidence penalty when modalities conflict.
    - FUSION_MISSING_FIELD_PENALTY: Penalty applied when fields are missing.
    - FUSION_TEMPORAL_DECAY_EXPONENT: Controls how fusion confidence decays with increasing temporal gap between inputs.
    - FUSION_AMBIGUITY_CONFIDENCE_THRESHOLD: Threshold below which results are flagged as ambiguous and may require confirmation.

Experiment settings:
    - TRIALS_PER_CONDITION: Number of trials per interaction mode.

Input configuration:
    - VOICE_INPUT: Voice input type ("mic" or "typed").
    - GESTURE_INPUT: Gesture input type ("webcam" or "buttons").
    - WHISPER_MODEL: Whisper model size used for speech recognition.

Gesture system:
    - GESTURE_TIMEOUT_SECONDS: Max time to wait for gesture input.
    - GESTURE_MODEL_PATH: Path to MediaPipe gesture recogniser model file.
    - GESTURE_OVERLAY_ENABLED: Whether to draw landmarks on webcam feed.
    - GESTURE_STABILITY_SECS: Time a gesture must be held to auto-confirm.

Logging:
- LOG_DIR: Directory where session logs are stored.

ROS 2 bridge:
    - ROS_DISPATCH_ENABLED: Whether to enable dispatching commands to the ROS 2 bridge
    - ROS_BRIDGE_URL: Base URL of the ROS 2 bridge (e.g., "http://stretch.local:5050")
"""

FUSION_WINDOW_SECONDS: float = 3.0
FUSION_VOICE_CONFIDENCE_THRESHOLD: float = 0.4
FUSION_GESTURE_ACTION_CONFIDENCE_THRESHOLD: float = 0.5
FUSION_GESTURE_LOCATION_CONFIDENCE_THRESHOLD: float = 0.5
FUSION_VOICE_BIAS_MARGIN: float = 0.15
FUSION_AGREEMENT_BONUS: float = 0.1
FUSION_CONFLICT_PENALTY: float = 0.1
FUSION_MISSING_FIELD_PENALTY: float = 0.05
FUSION_TEMPORAL_DECAY_EXPONENT: float = 1.0
FUSION_AMBIGUITY_CONFIDENCE_THRESHOLD: float = 0.15

TRIALS_PER_CONDITION: int = 10

VOICE_INPUT: str = "mic"
GESTURE_INPUT: str = "webcam"
WHISPER_MODEL: str = "base"

GESTURE_TIMEOUT_SECONDS: float = 10.0
GESTURE_MODEL_PATH: str = "gesture/gesture_recognizer.task"
GESTURE_OVERLAY_ENABLED: bool = True
GESTURE_STABILITY_SECS: float = 1.0

LOG_DIR: str = "logs"

ROS_DISPATCH_ENABLED: bool = False
ROS_BRIDGE_URL: str = "http://stretch.local:5050"  # TODO: update with actual robot hostname or IP address
