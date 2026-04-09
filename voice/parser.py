"""
Text-to-intent parser for voice module.

Converts a raw text string into a RobotCommand using keyword matching. Implements a rule-based approach with synonym dictionaries for actions, objects, and locations.
"""

from models import Action, Location, Mode, ObjectName, RobotCommand

# Synonym maps
ACTION_SYNONYMS: dict[str, Action] = {
    # pick
    "pick": Action.pick,
    "grab": Action.pick,
    "take": Action.pick,
    "lift": Action.pick,
    "get": Action.pick,
    "fetch": Action.pick,
    # place
    "place": Action.place,
    "put": Action.place,
    "drop": Action.place,
    "set": Action.place,
    "release": Action.place,
    # move
    "move": Action.move,
    "bring": Action.move,
    "shift": Action.move,
    "push": Action.move,
    "slide": Action.move,
    # stop
    "stop": Action.stop,
    "halt": Action.stop,
    "freeze": Action.stop,
    "pause": Action.stop,
    # cancel
    "cancel": Action.cancel,
    "abort": Action.cancel,
    "reset": Action.cancel,
    "undo": Action.cancel,
}

OBJECT_SYNONYMS: dict[str, ObjectName] = {
    "red cube": ObjectName.red_cube,
    "red block": ObjectName.red_cube,
    "red box": ObjectName.red_cube,
    "blue cube": ObjectName.blue_cube,
    "blue block": ObjectName.blue_cube,
    "blue box": ObjectName.blue_cube,
    "bottle": ObjectName.bottle,
    "flask": ObjectName.bottle,
    "container": ObjectName.bottle,
    "strawberry": ObjectName.strawberry,
    "strawberries": ObjectName.strawberry,
    "orange": ObjectName.orange,
    "fruit": ObjectName.fruit,
}

LOCATION_SYNONYMS: dict[str, Location] = {
    "left": Location.left,
    "left side": Location.left,
    "on the left": Location.left,
    "right": Location.right,
    "right side": Location.right,
    "on the right": Location.right,
    "table": Location.table,
    "on the table": Location.table,
    "desk": Location.table,
    "bin": Location.bin,
    "in the bin": Location.bin,
    "trash": Location.bin,
    "basket": Location.bin,
}


def _match_longest(text: str, synonyms: dict[str, object]) -> object | None:
    """
    Return value for the longest matching key in text.

    Args:
        text: Normalised input string.
        synonyms: Mapping of phrases to values.

    Returns:
        Matched value or None.
    """

    best_key = None
    best_len = 0

    for key in synonyms:
        if key in text and len(key) > best_len:
            best_key = key
            best_len = len(key)

    return synonyms[best_key] if best_key is not None else None


def parse_text_to_intent(text: str) -> RobotCommand:
    """
    Convert text into a RobotCommand.

    Uses keyword matching with longest-match priority. Missing fields remain None.

    Confidence:
        action -> +0.5
        object -> +0.25
        location -> +0.25

    Args:
        text: Raw input string.

    Returns:
        RobotCommand with extracted fields and confidence scores.
    """

    normalised = text.lower().strip()

    action: Action | None = _match_longest(normalised, ACTION_SYNONYMS)
    obj: ObjectName | None = _match_longest(normalised, OBJECT_SYNONYMS)
    location: Location | None = _match_longest(normalised, LOCATION_SYNONYMS)

    # Binary field-levl confidence (matched or not)
    action_conf = 1.0 if action is not None else 0.0
    object_conf = 1.0 if obj is not None else 0.0
    location_conf = 1.0 if location is not None else 0.0

    # Overall confidence based on completeness
    if action is None:
        confidence = 0.0
    else:
        confidence = (
            0.5
            + (0.25 if obj is not None else 0.0)
            + (0.25 if location is not None else 0.0)
        )

    return RobotCommand(
        mode=Mode.voice,
        action=action,
        object=obj,
        location=location,
        confidence=confidence,
        action_confidence=action_conf,
        object_confidence=object_conf,
        location_confidence=location_conf,
    )
