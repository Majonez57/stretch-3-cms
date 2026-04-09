"""
Shared enums and dataclasses for the multimodal HRI command processing.

This module defines the core data structures used across the system, including commmand representations, trial definitions, and fusion results. These models provide a common interface between input parsing, fusion, experiment logging, and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Action(str, Enum):
    """
    Robot actions recognised by the system.

    Attributes:
        pick: Command to pick up an object.
        place: Command to place an object down.
        move: Command to move to a location without an object.
        stop: Command to immediately stop all actions.
        cancel: Command to cancel the current action and return to idle.
    """

    pick = "pick"
    place = "place"
    move = "move"
    stop = "stop"
    cancel = "cancel"


class ObjectName(str, Enum):
    """
    Objects the robot can interact with.

    Attributes:
        red_cube: A red cube object.
        blue_cube: A blue cube object.
        bottle: A bottle object.
        strawberry: A strawberry object.
        orange: An orange object.
        fruit: An unspecified fruit object.
        none: No object specified or not applicable.
    """

    red_cube = "red_cube"
    blue_cube = "blue_cube"
    bottle = "bottle"
    strawberry = "strawberry"
    orange = "orange"
    fruit = "fruit"
    none = "none"


class Location(str, Enum):
    """
    Spatial locations used in robot commands.

    Attributes:
        left: The left side of the workspace.
        right: The right side of the workspace.
        table: The central table area.
        bin: The storage bin area.
        none: No location specified or not applicable.
    """

    left = "left"
    right = "right"
    table = "table"
    bin = "bin"
    none = "none"


class Mode(str, Enum):
    """
    Input modality that produced a command.

    Attributes:
        voice: Command produced from voice input.
        gesture: Command produced from gesture input.
        multimodal: Command produced from fusion of voice and gesture.
    """

    voice = "voice"
    gesture = "gesture"
    multimodal = "multimodal"


# Required fields for each action type
ACTION_FIELD_REQUIREMENTS: dict[str, list[str]] = {
    Action.pick: ["object", "location"],
    Action.place: ["location"],
    Action.move: ["location"],
    Action.stop: [],
    Action.cancel: [],
}


@dataclass
class RobotCommand:
    """
    Interpreted robot command from any input mode.

    Attributes:
        mode: Modality that produced the command.
        action: Resolved action, or None if unresolved.
        object: Resolved target object, or None if not applicable.
        location: Resolved target location, or None if not applicable.
        confidence: Overall command confidence.
        latency_ms: Time from input to command completion.
        timestamp: ISO timestamp of command creation.
        action_confidence: Confidence for the action field.
        object_confidence: Confidence for the object field.
        location_confidence: Confidence for the location field.
    """

    mode: Mode
    action: Optional[Action] = None
    object: Optional[ObjectName] = None
    location: Optional[Location] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: Optional[str] = None
    action_confidence: float = 0.0
    object_confidence: float = 0.0
    location_confidence: float = 0.0


@dataclass
class TrialDefinition:
    """
    Predefined trial presented to a participant during an experiment.

    Attributes:
        trial_id: Unique identifier for the trial.
        condition: Input modality for the trial.
        expected_action: The action expected from the participant.
        expected_object: The object expected from the participant.
        expected_location: The location expected from the participant.
        prompt_text: The instruction shown to the participant for this trial.
    """

    trial_id: int
    condition: Mode
    expected_action: Action
    expected_object: ObjectName
    expected_location: Location
    prompt_text: str


@dataclass
class TrialResult:
    """
    Logged result for one completed trial.

    Attributes:
        participant_id: Unique identifier for the participant.
        condition: Input modality used in the trial.
        trial_id: Unique identifier for the trial.
        expected_action: The action expected from the participant.
        expected_object: The object expected from the participant.
        expected_location: The location expected from the participant.
        predicted_action: The action predicted by the system, or None if unresolved.
        predicted_object: The object predicted by the system, or None if unresolved/not applicable.
        predicted_location: The location predicted by the system, or None if unresolved/not applicable.
        correct: True if the predicted command matches the expected command.
        latency_ms: Time from trial start to command completion in milliseconds.
        correction_count: Number of times the participant corrected their input during the trial.
        conflict_flag: True if there was a conflict between voice and gesture inputs (for multimodal trials).
        voice_timestamp: ISO timestamp of the voice input event, or None if not applicable.
        gesture_timestamp: ISO timestamp of the gesture input event, or None if not applicable.
        fusion_within_window: True if voice and gesture inputs arrived within the fusion window (for multimodal trials), or None if not applicable.
        timestamp: ISO timestamp when the trial result was recorded.
    """

    participant_id: str
    condition: Mode
    trial_id: int
    expected_action: Action
    expected_object: ObjectName
    expected_location: Location
    predicted_action: Optional[Action]
    predicted_object: Optional[ObjectName]
    predicted_location: Optional[Location]
    correct: bool
    latency_ms: float
    correction_count: int
    conflict_flag: bool
    voice_timestamp: Optional[str]
    gesture_timestamp: Optional[str]
    fusion_within_window: Optional[bool]
    timestamp: str
    confidence: float = 0.0


@dataclass
class FusionResult:
    """
    Fusion output and supporting metadata.

    Attributes:
        command: Resolved command after fusion.
        conflict_fields: Fields where modalities disagreed.
        voice_values: Raw voice values before fusion.
        gesture_values: Raw gesture values before fusion.
        within_window: Whether inputs fell within the fusion window.
        voice_timestamp: Unix timestamp of the voice event.
        gesture_timestamp: Unix timestamp of the gesture event.
        field_source: Source of each resolved field.
        used_voice: Whether voice contributed to the result.
        used_gesture: Whether gesture contributed to the result.
        multimodal_contribution_count: Number of jointly provided fields.
        needs_confirmation: Whether the result is ambiguous.
        ambiguity_reason: Explanation of the ambiguity, if any.
        diagnostics: Additional fusion diagnostics.
    """

    command: RobotCommand
    conflict_fields: list[str] = field(default_factory=list)
    voice_values: dict = field(default_factory=dict)
    gesture_values: dict = field(default_factory=dict)
    within_window: bool = True
    voice_timestamp: Optional[float] = None
    gesture_timestamp: Optional[float] = None
    field_source: dict[str, str] = field(default_factory=dict)
    used_voice: bool = False
    used_gesture: bool = False
    multimodal_contribution_count: int = 0
    needs_confirmation: bool = False
    ambiguity_reason: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)
