"""
Experiment session controller for trial-based evaluation.

Manages trial progression, result recording, and summary generation for a single particpant session.
"""

from datetime import datetime, timezone
from typing import Optional

from experiments.trials import get_ordered_trials
from models import (
    Action,
    Location,
    Mode,
    ObjectName,
    TrialDefinition,
    TrialResult,
)


class ExperimentRunner:
    """
    Manages trial progression and results for a session.

    Attributes:
        participant_id: Identifier for the participant.
        _trials: Ordered list of trials.
        _index: Current trial index.
        _results: Recorded trial results.
    """

    def __init__(
        self,
        participant_id: str = "system",
        trials: Optional[list[TrialDefinition]] = None,
    ) -> None:
        """
        Initialise a session runner.

        Args:
            participant_id: Participant identifier.
            trials: Optional predefined trial list.
        """

        self.participant_id = participant_id
        self._trials = (
            trials if trials is not None else get_ordered_trials(participant_id)
        )
        self._index: int = 0
        self._results: list[TrialResult] = []

    def get_current_trial(self) -> Optional[TrialDefinition]:
        """
        Trial navigation method to get the current trial, or None if the session is complete.

        Returns:
            Current TrialDefinition, or None.
        """

        if self._index < len(self._trials):
            return self._trials[self._index]

        return None

    def has_next(self) -> bool:
        """
        Trial navigation method to check if there are more trials after the current one.

        Returns:
            True if another trial exists, otherwise False.
        """

        return self._index < len(self._trials) - 1

    def advance(self) -> None:
        """
        Trial navigation method to advance to the next trial.

        Trial Navigation.
        """
        if self._index < len(self._trials):
            self._index += 1

    def submit_result(
        self,
        predicted_action: Optional[Action],
        predicted_object: Optional[ObjectName],
        predicted_location: Optional[Location],
        latency_ms: float,
        correction_count: int = 0,
        conflict_flag: bool = False,
        voice_timestamp: Optional[str] = None,
        gesture_timestamp: Optional[str] = None,
        fusion_within_window: Optional[bool] = None,
        confidence: float = 0.0,
    ) -> TrialResult:
        """
        Record the result for the current trial.

        Args:
            predicted_object: Predicted object.
            predicted_location: Predicted location.
            latency_ms: Response latency.
            correction_count: Number of retries.
            conflict_flag: Whether fusion conflict occurred.
            voice_timestamp: Voice input timestamp.
            gesture_timestamp: Gesture input timestamp.
            fusion_within_window: Whether fusion timing was valid.
            confidence: Command confidence score.

        Returns:
            Recorded TrialResult.

        Raises:
            RuntimeError: If no active trial.
        """

        trial = self.get_current_trial()
        if trial is None:
            raise RuntimeError("No current trial — session is complete.")

        correct = _is_correct(
            trial, predicted_action, predicted_object, predicted_location
        )

        result = TrialResult(
            participant_id=self.participant_id,
            condition=trial.condition,
            trial_id=trial.trial_id,
            expected_action=trial.expected_action,
            expected_object=trial.expected_object,
            expected_location=trial.expected_location,
            predicted_action=predicted_action,
            predicted_object=predicted_object,
            predicted_location=predicted_location,
            correct=correct,
            latency_ms=latency_ms,
            correction_count=correction_count,
            conflict_flag=conflict_flag,
            voice_timestamp=voice_timestamp,
            gesture_timestamp=gesture_timestamp,
            fusion_within_window=fusion_within_window,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
        )
        self._results.append(result)

        return result

    def progress(self) -> tuple[int, int]:
        """
        Return the current progress of the session.

        Returns:
            Tuple of (current_index, total_trials).
        """

        return self._index, len(self._trials)

    def get_summary(self) -> dict:
        """
        Gets the aggregated session results.

        Returns:
            Dict with a summary containing overall accuracy and per-condition stats.
        """

        completed = len(self._results)

        if completed == 0:
            return {
                "participant_id": self.participant_id,
                "total_trials": len(self._trials),
                "completed_trials": 0,
                "overall_accuracy": 0.0,
                "by_condition": {},
            }

        by_condition: dict[str, dict] = {}

        for result in self._results:
            key = result.condition.value
            if key not in by_condition:
                by_condition[key] = {"trials": 0, "correct": 0}

            by_condition[key]["trials"] += 1
            by_condition[key]["correct"] += int(result.correct)

        for stats in by_condition.values():
            stats["accuracy"] = stats["correct"] / stats["trials"]

        total_correct = sum(r.correct for r in self._results)

        return {
            "participant_id": self.participant_id,
            "total_trials": len(self._trials),
            "completed_trials": completed,
            "overall_accuracy": total_correct / completed,
            "by_condition": by_condition,
        }


def _is_correct(
    trial: TrialDefinition,
    predicted_action: Optional[Action],
    predicted_object: Optional[ObjectName],
    predicted_location: Optional[Location],
) -> bool:
    """
    Helper method to check if prediction matches required fields for the action.

    Args:
        trial: Trial definition with expected values.
        predicted_action: Predicted action.
        predicted_object: Predicted object.
        predicted_location: Predicted location.

    Returns:
        True if prediction is correct, otherwise False.
    """

    if predicted_action != trial.expected_action:
        return False

    action = trial.expected_action

    # stop - action only
    if action in (Action.stop, Action.cancel):
        return True

    # place — matching location
    if action in (Action.place, Action.move):
        return predicted_location == trial.expected_location

    # pick - matching object and location
    if action == Action.pick:
        return (
            predicted_object == trial.expected_object
            and predicted_location == trial.expected_location
        )

    return False
