"""
Main Streamlit application for the multimodal HRI command hub.

Two interface modes:
  - Live Interaction Mode: primary product interface for freeform command input
  - Trial / Experiment Mode: evaluation layer with prompts, correctness, and logging

Manages UI flow, session state, multimodal input handling, and experiment logging.
"""

import time
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

import config
from experiments.runner import ExperimentRunner, _is_correct
from fusion.fuser import fuse_inputs
from gesture.detector import GestureResult, infer_hand_location
from gesture.mapper import CONFIRM_GESTURE, map_gesture_to_intent
from models import (
    Action,
    FusionResult,
    Location,
    Mode,
    RobotCommand,
    TrialDefinition,
)
from trial_logger.logger import SessionLogger
from ui.components import (
    render_command_output,
    render_command_panel,
    render_header,
    render_mode_badge,
    render_progress,
    render_session_summary,
    render_trial_prompt,
)
from ros2.dispatch import dispatch_if_enabled
from voice.parser import parse_text_to_intent
from voice.validation import validate_command

# Button label to gesture label
ACTION_GESTURE_OPTIONS: dict[str, str] = {
    "✊  Closed Fist — pick": "Closed_Fist",
    "🤚  Open Palm — stop": "Open_Palm",
    "✌️  Victory — place": "Victory",
}

# Button label to location
LOCATION_OPTIONS: dict[str, Location] = {
    "👈  Left": Location.left,
    "👉  Right": Location.right,
}

# Valid action gestures for webcam input
_ACTION_GESTURES: set[str] = {"Closed_Fist", "Open_Palm", "Victory"}


def _get_gesture_processor_class(overlay_enabled: bool = True):
    """
    Return the shared webcam gesture processor class.

    Creates a VideoProcessorBase subclass for use with streamlit-webrtc.
    The processor performs gesture detection, location inference, and
    optional overlay rendering.

    Args:
        overlay_enabled: Initial overlay state.

    Returns:
        GestureProcessor class for webrtc_streamer.
    """

    import threading

    from streamlit_webrtc import VideoProcessorBase

    from gesture.detector import detect_gesture_from_frame, infer_hand_location

    # Fixed hand landmark connections for overlay drawing
    _HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    def _draw_overlay(img, result: GestureResult) -> None:
        """
        Draw hand landmarks and connections.

        Args:
            img: Frame image in BGR format.
            result: Detected gesture result.
        """

        import cv2

        h, w = img.shape[:2]

        # Draw hand connections
        for start_idx, end_idx in _HAND_CONNECTIONS:
            start = result.hand_landmarks[start_idx]
            end = result.hand_landmarks[end_idx]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(img, pt1, pt2, (0, 200, 200), 2)

        # Draw landmark points
        for lm in result.hand_landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(img, pt, 4, (255, 255, 255), -1)

    class GestureProcessor(VideoProcessorBase):
        """
        Process webcam frames for gesture interaction.

        Attributes:
            gesture_result: Latest detected gesture result.
            inferred_location: Inferred left/right location from hand position.
            inferred_location_confidence: Confidence of inferred location.
            last_action_result: Last detected action gesture (for confirmation).
            thumb_up_confirm: True if a Thumb_Up gesture was detected.
            detection_ts: Timestamp of last detected gesture.
            overlay_enabled: Whether to draw overlay on frames.
            lock: Thread lock for safe access to shared state.
        """

        def __init__(self):
            """
            Initialise the processor state.
            """

            self.gesture_result: Optional[GestureResult] = None
            self.inferred_location: Optional[Location] = None
            self.inferred_location_confidence: float = 0.0
            self.last_action_result: Optional[GestureResult] = None
            self.thumb_up_confirm: bool = False
            self.detection_ts: Optional[float] = None
            self.overlay_enabled: bool = overlay_enabled
            self.lock = threading.Lock()

        def recv(self, frame):
            """
            Process one webcam frame.

            Args:
                frame: Incoming WebRTC video frame.

            Returns:
                Processed video frame with optional overlay.
            """

            import av
            import cv2

            # RGB to BGR
            img = frame.to_ndarray(format="rgb24")
            img = img[:, :, ::-1].copy()

            result = detect_gesture_from_frame(img)

            with self.lock:
                self.gesture_result = result
                if result is not None:
                    # Infer location and confidence from hand position
                    (
                        self.inferred_location,
                        self.inferred_location_confidence,
                    ) = infer_hand_location(result.hand_landmarks)
                    self.detection_ts = time.time()

                    # Cache the last action gesture for Thumb_Up confirmation
                    if result.gesture_label in _ACTION_GESTURES:
                        self.last_action_result = result
                        self.thumb_up_confirm = False
                    elif result.gesture_label == CONFIRM_GESTURE:
                        self.thumb_up_confirm = True

                do_overlay = self.overlay_enabled

            if do_overlay and result is not None:
                _draw_overlay(img, result)

            # Mirror the displayed feed
            img = cv2.flip(img, 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return GestureProcessor


def _render_gesture_status(
    result: Optional[GestureResult],
    show_location: bool = False,
    inferred_loc: Optional[Location] = None,
) -> None:
    """
    Display gesture detection status below the video feed.

    Args:
        result: Latest gesture result, or None if no hand detected.
        show_location: Whether to display location instead of confidence.
        inferred_loc: Inferred location when show_location is True.
    """

    hand_detected = result is not None
    has_gesture = result is not None and result.gesture_label not in (
        None,
        "None",
        "",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hand", "detected" if hand_detected else "none")
    with col2:
        st.metric("Gesture", result.gesture_label if has_gesture else "—")
    with col3:
        if show_location:
            st.metric("Location", inferred_loc.value if inferred_loc else "—")
        else:
            st.metric(
                "Confidence", f"{result.confidence:.0%}" if has_gesture else "—"
            )


def _debug_pre_fusion(
    voice_cmd: RobotCommand,
    missing: list[str],
    pre_loc: Optional[Location],
    gesture_ts: Optional[float],
    gesture_confidence: float,
    location_confidence: float,
) -> None:
    """
    Display debug information for fusion inputs.

    Args:
        voice_cmd: Parsed voice command.
        missing: List of missing fields after validation.
        pre_loc: Gesture-derived location, if available.
        gesture_ts: Timestamp of gesture detection.
        gesture_confidence: Confidence of gesture action.
        location_confidence: Confidence of inferred location.
    """

    with st.expander("Debug: Fusion State", expanded=False):
        st.markdown(
            f"**Voice parsed:** action={voice_cmd.action}, "
            f"object={voice_cmd.object}, location={voice_cmd.location}"
        )
        st.markdown(
            f"**Voice confidence:** overall={voice_cmd.confidence:.2f}, "
            f"action={voice_cmd.action_confidence:.2f}, "
            f"object={voice_cmd.object_confidence:.2f}, "
            f"location={voice_cmd.location_confidence:.2f}"
        )
        st.markdown(f"**Missing fields:** {missing if missing else 'none'}")
        st.markdown(
            f"**Gesture captured:** {'yes' if pre_loc is not None else 'NO'}"
        )
        if pre_loc is not None:
            st.markdown(
                f"**Gesture location:** {pre_loc.value}, "
                f"confidence={location_confidence:.2f}, "
                f"gesture_ts={gesture_ts}"
            )
            st.markdown(
                f"**Gesture action confidence:** {gesture_confidence:.2f}"
            )
        else:
            st.markdown("**Gesture:** no hand detected during voice window")
        st.markdown(
            f"**Reset state:** live_cmd_gen={st.session_state.get('live_cmd_gen', 0)}, "
            f"live_multimodal_step={st.session_state.get('live_multimodal_step', 0)}"
        )


def _read_processor_location(ctx) -> Optional[Location]:
    """
    Return the latest inferred location from the webcam processor.

    Args:
        ctx: webrtc_streamer context.

    Returns:
        Latest inferred location, or None.
    """

    if ctx is None or ctx.video_processor is None:
        return None
    with ctx.video_processor.lock:
        return ctx.video_processor.inferred_location


def _read_gesture_state_for_fusion(
    ctx,
    voice_start_ts: Optional[float],
) -> tuple[Optional[Location], Optional[float], Optional[GestureResult], float]:
    """
    Read gesture data for multimodal fusion.

    Only returns data if it was captured during the current voice input window.

    Args:
        ctx: webrtc_streamer context.
        voice_start_ts: Timestamp when voice input began.

    Returns:
        Tuple of (location, detection_ts, gesture_result, location_confidence).
        Returns (None, None, None, 0.0) if no valid gesture is available.
    """

    if ctx is None or ctx.video_processor is None:
        return None, None, None, 0.0

    with ctx.video_processor.lock:
        loc = ctx.video_processor.inferred_location
        loc_conf = ctx.video_processor.inferred_location_confidence
        det_ts = ctx.video_processor.detection_ts
        result = ctx.video_processor.gesture_result

    if loc is None or det_ts is None or result is None:
        return None, None, None, 0.0

    if voice_start_ts is not None and det_ts < voice_start_ts:
        return None, None, None, 0.0

    return loc, det_ts, result, loc_conf


def _init_state() -> None:
    """
    Initialise Streamlit session state on first load.
    """
    defaults: dict = {
        "phase": "setup",
        # Session objects
        "runner": None,
        "logger": None,
        "participant_id": "",
        # Timing
        "trial_start_s": None,
        # Gesture mode state
        "gesture_step": 0,  # 0 = action, 1 = location
        "gesture_action": None,
        # Multimodal trial state
        "multimodal_step": 0,  # 0 = voice, 1 = gesture
        "voice_cmd": None,
        "voice_ts": None,
        "voice_step_start_ts": None,
        # Retry tracking
        "correction_count": 0,
        # Pending trial result
        "pending_cmd": None,
        "pending_fusion": None,
        "pending_conflict_flag": False,
        "pending_voice_ts": None,
        "pending_gesture_ts": None,
        "pending_fusion_within_window": None,
        # Live mode
        "live_mode": None,
        "live_cmd": None,
        "live_cmd_dispatched": False,
        "live_gesture_step": 0,
        "live_gesture_action": None,
        "live_multimodal_step": 0,
        "live_voice_cmd": None,
        "live_voice_ts": None,
        "live_voice_start_ts": None,
        # Widget reset counter
        "live_cmd_gen": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_trial_setup() -> None:
    """
    Render participant setup screen and start experiment.
    """
    st.subheader("Experiment Setup")
    st.caption("Enter a participant ID and start the session.")

    participant_id = st.text_input(
        "Participant ID",
        value="P01",
        placeholder="e.g. P01",
        help="Leave blank to use system evaluation mode.",
    )

    if st.button("Start Experiment", type="primary"):
        pid = participant_id.strip() or "system"

        st.session_state.runner = ExperimentRunner(participant_id=pid)
        st.session_state.logger = SessionLogger(participant_id=pid)
        st.session_state.participant_id = pid

        st.session_state.phase = "trial_input"
        st.session_state.trial_start_s = time.time()

        st.rerun()


def _run_live_page() -> None:
    """
    Render the live interaction dashboard
    """

    hdr_col, mode_col = st.columns([2, 3])
    with hdr_col:
        st.markdown("### Live Interaction Mode")
    with mode_col:
        mode = st.radio(
            "Mode",
            options=["Voice", "Gesture", "Multimodal"],
            horizontal=True,
            key="live_mode_selector",
            label_visibility="collapsed",
        )

    mode_lower = mode.lower()
    if st.session_state.live_mode != mode_lower:
        st.session_state.live_mode = mode_lower
        _reset_live_state()

    col_left, col_right = st.columns([3.2, 1.8], gap="large")

    with col_right:
        st.markdown(
            '<div class="live-right-panel"></div>', unsafe_allow_html=True
        )
        st.markdown("**Status**")

    if mode_lower == "voice":
        _live_voice(col_left, col_right)
    elif mode_lower == "gesture":
        _live_gesture(col_left, col_right)
    elif mode_lower == "multimodal":
        _live_multimodal(col_left, col_right)

    with col_right:
        st.markdown("---")
        if st.session_state.live_cmd is not None:
            if not st.session_state.live_cmd_dispatched:
                dispatch_if_enabled(st.session_state.live_cmd)
                st.session_state.live_cmd_dispatched = True
            render_command_panel(st.session_state.live_cmd)
        else:
            st.markdown("**Recognised Command**")
            st.caption("Waiting for input…")

    if st.session_state.live_cmd is not None:
        with col_left:
            if st.button("New Command", type="primary"):
                _reset_live_state()
                st.rerun()


def _reset_live_state() -> None:
    """
    Reset live interaction state for a new command.
    """

    st.session_state.live_cmd = None
    st.session_state.live_cmd_dispatched = False
    st.session_state.live_gesture_step = 0
    st.session_state.live_gesture_action = None
    st.session_state.live_multimodal_step = 0
    st.session_state.live_voice_cmd = None
    st.session_state.live_voice_ts = None
    st.session_state.live_voice_start_ts = None
    st.session_state.live_cmd_gen = st.session_state.get("live_cmd_gen", 0) + 1
    st.session_state.pop("_live_gesture_action_cache", None)
    st.session_state.pop("_live_gesture_loc_cache", None)


def _live_voice(col_left, col_right) -> None:
    """
    Handle live voice input (typed or microphone).

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    if config.VOICE_INPUT == "mic":
        _live_voice_mic(col_left, col_right)
    else:
        _live_voice_typed(col_left, col_right)


def _live_voice_typed(col_left, col_right) -> None:
    """
    Handle typed voice input.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    gen = st.session_state.get("live_cmd_gen", 0)
    with col_left:
        with st.form(f"live_voice_form_{gen}"):
            text = st.text_input(
                "Command", placeholder="e.g. pick the red cube left"
            )
            submitted = st.form_submit_button("Submit Command", type="primary")

    if submitted and text.strip():
        cmd = parse_text_to_intent(text.strip())
        cmd.mode = Mode.voice
        cmd.latency_ms = 0.0
        cmd.timestamp = _iso_now()
        st.session_state.live_cmd = cmd
        st.rerun()
    elif submitted:
        with col_left:
            st.warning("Please enter a command before submitting.")


def _live_voice_mic(col_left, col_right) -> None:
    """
    Handle microphone voice inut using audio recording.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """
    with col_left:
        gen = st.session_state.get("live_cmd_gen", 0)
        audio_bytes = st.audio_input("Record command", key=f"live_audio_{gen}")

    if audio_bytes is not None and st.session_state.live_cmd is None:
        with col_left:
            with st.spinner("Transcribing…"):
                from voice.speech import transcribe_audio_bytes

                text = transcribe_audio_bytes(audio_bytes.getvalue())
            if text:
                st.write(f"**Heard:** {text}")
                cmd = parse_text_to_intent(text)
                cmd.mode = Mode.voice
                cmd.latency_ms = 0.0
                cmd.timestamp = _iso_now()
                st.session_state.live_cmd = cmd
            else:
                st.warning("No speech detected. Please try again.")


def _live_gesture(col_left, col_right) -> None:
    """
    Handle live gesture input (webcam or buttons).

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    if config.GESTURE_INPUT == "webcam":
        _live_gesture_webcam(col_left, col_right)
    else:
        _live_gesture_buttons(col_left, col_right)


def _live_gesture_buttons(col_left, col_right) -> None:
    """
    Handle button gesture input (two-step interaction).

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    step = st.session_state.live_gesture_step

    if step == 0:
        with col_left:
            with st.form("live_gesture_action_form"):
                choice = st.radio(
                    "Action gesture",
                    options=list(ACTION_GESTURE_OPTIONS.keys()),
                    label_visibility="collapsed",
                )
                confirmed = st.form_submit_button(
                    "Confirm Action Gesture", type="primary"
                )

        if confirmed:
            gesture_name = ACTION_GESTURE_OPTIONS[choice]
            intent = map_gesture_to_intent(gesture_name)
            action: Action = intent["action"]
            st.session_state.live_gesture_action = action

            if action in (Action.stop, Action.cancel):
                cmd = RobotCommand(
                    mode=Mode.gesture,
                    action=action,
                    confidence=1.0,
                    timestamp=_iso_now(),
                )
                st.session_state.live_cmd = cmd
                st.rerun()
            else:
                st.session_state.live_gesture_step = 1
                st.rerun()
    else:
        action = st.session_state.live_gesture_action
        with col_left:
            st.caption(f"Action: {action.value} — select location:")
            with st.form("live_gesture_location_form"):
                choice = st.radio(
                    "Location",
                    options=list(LOCATION_OPTIONS.keys()),
                    label_visibility="collapsed",
                )
                confirmed = st.form_submit_button(
                    "Confirm Location", type="primary"
                )

        if confirmed:
            location = LOCATION_OPTIONS[choice]
            cmd = RobotCommand(
                mode=Mode.gesture,
                action=action,
                location=location,
                confidence=1.0,
                timestamp=_iso_now(),
            )
            st.session_state.live_cmd = cmd
            st.rerun()


def _live_gesture_webcam(col_left, col_right) -> None:
    """
    Handle webcam gesture input using streamlit-webrtc.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    try:
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        with col_left:
            st.error(
                "streamlit-webrtc is not installed. Run: pip install streamlit-webrtc"
            )
        return

    step = st.session_state.live_gesture_step

    with col_left:
        overlay_on = st.checkbox(
            "Show hand overlay",
            value=config.GESTURE_OVERLAY_ENABLED,
            key="live_gesture_overlay",
        )
        GestureProcessor = _get_gesture_processor_class(
            overlay_enabled=overlay_on
        )

        if step == 0:
            st.caption(
                "Step 1/2 — Show a gesture: ✊ pick · 🤚 stop · ✌️ place · Hold to auto-capture or click Confirm."
            )
        else:
            action = st.session_state.live_gesture_action
            st.caption(
                f"Step 2/2 — Action: {action.value}. Position hand left or right of centre, then click Confirm."
            )

        ctx = webrtc_streamer(
            key=f"live_gesture_{step}",
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            translations={"stop": "Confirm Gesture"},
        )

    if not ctx.state.playing:
        if step == 0:
            cached = st.session_state.pop("_live_gesture_action_cache", None)
            if cached is not None:
                if cached.gesture_label not in _ACTION_GESTURES:
                    with col_left:
                        st.warning(
                            "No action gesture detected. Show ✊ Closed Fist, 🤚 Open Palm, or ✌️ Victory."
                        )
                else:
                    intent = map_gesture_to_intent(cached.gesture_label)
                    action = intent["action"]
                    st.session_state.live_gesture_action = action
                    if action in (Action.stop, Action.cancel):
                        cmd = RobotCommand(
                            mode=Mode.gesture,
                            action=action,
                            confidence=cached.confidence,
                            timestamp=_iso_now(),
                        )
                        st.session_state.live_cmd = cmd
                        st.rerun()
                    else:
                        st.session_state.live_gesture_step = 1
                        st.rerun()
        else:
            cached = st.session_state.pop("_live_gesture_loc_cache", None)
            if cached is not None:
                inferred_loc, confidence = cached
                if inferred_loc is not None:
                    cmd = RobotCommand(
                        mode=Mode.gesture,
                        action=st.session_state.live_gesture_action,
                        location=inferred_loc,
                        confidence=confidence,
                        timestamp=_iso_now(),
                    )
                    st.session_state.live_cmd = cmd
                    st.rerun()

    if ctx.video_processor:
        ctx.video_processor.overlay_enabled = overlay_on

        # Action step to detect stable gesture
        if step == 0:
            status_placeholder = col_right.empty()

            stable_label: Optional[str] = None
            stable_since: Optional[float] = None
            while ctx.state.playing:
                with ctx.video_processor.lock:
                    gesture_result = ctx.video_processor.gesture_result
                    last_action = ctx.video_processor.last_action_result
                    thumb_up = ctx.video_processor.thumb_up_confirm

                st.session_state["_live_gesture_action_cache"] = (
                    last_action
                    if (thumb_up and last_action)
                    else gesture_result
                )

                current_label = (
                    gesture_result.gesture_label
                    if gesture_result
                    and gesture_result.gesture_label in _ACTION_GESTURES
                    else None
                )
                if current_label != stable_label:
                    stable_label = current_label
                    stable_since = time.time() if current_label else None

                if (
                    stable_label
                    and stable_since
                    and (time.time() - stable_since)
                    >= config.GESTURE_STABILITY_SECS
                ):
                    intent = map_gesture_to_intent(stable_label)
                    action = intent["action"]
                    st.session_state.live_gesture_action = action
                    if action in (Action.stop, Action.cancel):
                        cmd = RobotCommand(
                            mode=Mode.gesture,
                            action=action,
                            confidence=(
                                gesture_result.confidence
                                if gesture_result
                                else 1.0
                            ),
                            timestamp=_iso_now(),
                        )
                        st.session_state.live_cmd = cmd
                    else:
                        st.session_state.live_gesture_step = 1
                    st.rerun()

                with status_placeholder.container():
                    _render_gesture_status(gesture_result)
                    if stable_label and stable_since:
                        elapsed = time.time() - stable_since
                        st.progress(
                            min(elapsed / config.GESTURE_STABILITY_SECS, 1.0),
                            text=f"Holding {stable_label}… {elapsed:.1f}s",
                        )
                    if (
                        gesture_result
                        and gesture_result.gesture_label == CONFIRM_GESTURE
                        and last_action
                    ):
                        st.info(
                            f"Thumb_Up — will confirm: **{last_action.gesture_label}**"
                        )
                time.sleep(0.1)

        # Location step to infer left/right from hand position
        else:
            status_placeholder = col_right.empty()

            while ctx.state.playing:
                with ctx.video_processor.lock:
                    inferred_loc = ctx.video_processor.inferred_location
                    gesture_result = ctx.video_processor.gesture_result

                st.session_state["_live_gesture_loc_cache"] = (
                    inferred_loc,
                    gesture_result.confidence if gesture_result else 1.0,
                )

                with status_placeholder.container():
                    _render_gesture_status(
                        gesture_result,
                        show_location=True,
                        inferred_loc=inferred_loc,
                    )
                time.sleep(0.1)


def _live_multimodal(col_left, col_right) -> None:
    """
    Handle live multimodal input flow.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    step = st.session_state.live_multimodal_step

    if step == 0:
        _live_multimodal_voice(col_left, col_right)
    else:
        _live_multimodal_gesture(col_left, col_right)


def _live_multimodal_voice(col_left, col_right) -> None:
    """
    Handle simultaneous voice and gesture input.

    Starts the voice interaction window and captures gesture state during that window for immediate fusion.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    # Start of the active voice window for simultaneous fusion
    if st.session_state.live_voice_start_ts is None:
        st.session_state.live_voice_start_ts = time.time()
    voice_start_ts = st.session_state.live_voice_start_ts

    with col_left:
        st.caption(
            "Speak and gesture simultaneously — hand position sets location."
        )

    mm_ctx = None
    if config.GESTURE_INPUT == "webcam":
        try:
            from streamlit_webrtc import webrtc_streamer

            GestureProcessor = _get_gesture_processor_class(
                overlay_enabled=config.GESTURE_OVERLAY_ENABLED
            )
            with col_left:
                mm_ctx = webrtc_streamer(
                    key="live_mm_gesture",
                    video_processor_factory=GestureProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                )

            # Show latest gesture state while voice input is active
            if mm_ctx.video_processor:
                with mm_ctx.video_processor.lock:
                    _cur_result = mm_ctx.video_processor.gesture_result
                    _cur_loc = mm_ctx.video_processor.inferred_location
                with col_right:
                    _render_gesture_status(
                        _cur_result, show_location=True, inferred_loc=_cur_loc
                    )
        except ImportError:
            pass

    gen = st.session_state.get("live_cmd_gen", 0)

    if config.VOICE_INPUT == "mic":
        with col_left:
            audio_bytes = st.audio_input(
                "Record command (position hand first)",
                key=f"live_mm_audio_{gen}",
            )

        # Guard against repeated processing across reruns
        if audio_bytes is not None and st.session_state.live_cmd is None:
            pre_loc, gesture_det_ts, gesture_result, loc_conf = (
                _read_gesture_state_for_fusion(mm_ctx, voice_start_ts)
            )

            # Capture submit time before transcription latency
            voice_submit_ts = time.time()

            with col_left:
                with st.spinner("Transcribing…"):
                    from voice.speech import transcribe_audio_bytes

                    text = transcribe_audio_bytes(audio_bytes.getvalue())
                if text:
                    st.write(f"**Heard:** {text}")
                    _live_multimodal_process_voice(
                        text,
                        pre_loc=pre_loc,
                        gesture_ts=gesture_det_ts,
                        gesture_confidence=(
                            gesture_result.confidence if gesture_result else 1.0
                        ),
                        location_confidence=loc_conf,
                        voice_submit_ts=voice_submit_ts,
                    )
                else:
                    st.warning("No speech detected. Please try again.")
    else:
        with col_left:
            with st.form(f"live_mm_voice_form_{gen}"):
                text = st.text_input(
                    "Voice command", placeholder="e.g. pick up the red cube"
                )
                submitted = st.form_submit_button(
                    "Submit Voice", type="primary"
                )

        if submitted and text.strip():
            pre_loc, gesture_det_ts, gesture_result, loc_conf = (
                _read_gesture_state_for_fusion(mm_ctx, voice_start_ts)
            )
            _live_multimodal_process_voice(
                text.strip(),
                pre_loc=pre_loc,
                gesture_ts=gesture_det_ts,
                gesture_confidence=(
                    gesture_result.confidence if gesture_result else 1.0
                ),
                location_confidence=loc_conf,
            )
        elif submitted:
            with col_left:
                st.warning("Please enter a command before submitting.")


def _live_multimodal_process_voice(
    text: str,
    pre_loc: Optional[Location] = None,
    gesture_ts: Optional[float] = None,
    gesture_confidence: float = 1.0,
    location_confidence: float = 1.0,
    voice_submit_ts: Optional[float] = None,
) -> None:
    """
    Parse voice input and apply multimodal fusion.

    Uses gesture data captured during the active voice window to fill missing location immediately. Falls back to a gesture step onlywhen needed.

    Args:
        text: Transcribed or typed voice input.
        pre_loc: Gesture-derived location, if available.
        gesture_ts: Gesture timestamp.
        gesture_confidence: Gesture confidence score.
        location_confidence: Confidence of inferred location.
        voice_submit_ts: Voice submission timestamp before transcription.
    """
    voice_cmd = parse_text_to_intent(text)

    # Use pre-transcription timestamp when available
    voice_ts = voice_submit_ts if voice_submit_ts is not None else time.time()

    st.session_state.live_voice_cmd = voice_cmd
    st.session_state.live_voice_ts = voice_ts

    missing = validate_command(voice_cmd)

    # Debug to show what was captured before fusion decisions
    _debug_pre_fusion(
        voice_cmd,
        missing,
        pre_loc,
        gesture_ts,
        gesture_confidence,
        location_confidence,
    )

    if voice_cmd.action in (Action.stop, Action.cancel) or not missing:
        voice_cmd.mode = Mode.multimodal
        voice_cmd.timestamp = _iso_now()
        st.session_state.live_cmd = voice_cmd
    elif voice_cmd.action is None:
        st.warning(
            "No action recognised. Try again, e.g. 'pick up the red cube'."
        )
    elif "location" in missing and pre_loc is not None:
        # Fuse immediately when location was captured during the voice window
        gesture_cmd = RobotCommand(
            mode=Mode.gesture,
            location=pre_loc,
            confidence=gesture_confidence,
            location_confidence=location_confidence,
        )

        g_ts = gesture_ts if gesture_ts is not None else voice_ts
        fusion_result = fuse_inputs(
            voice_cmd=voice_cmd,
            gesture_cmd=gesture_cmd,
            voice_ts=voice_ts,
            gesture_ts=g_ts,
        )

        if fusion_result.within_window:
            final_cmd = fusion_result.command
            final_cmd.timestamp = _iso_now()
            st.session_state.live_cmd = final_cmd
        else:
            if missing == ["location"]:
                st.info(
                    "Gesture was outside the time window — please gesture the location."
                )
                st.session_state.live_multimodal_step = 1
                st.rerun()
            else:
                st.warning(
                    f"Missing: {', '.join(missing)}. Please say the full command again."
                )
    elif missing == ["location"]:
        st.info("Location missing — please gesture left or right.")
        st.session_state.live_multimodal_step = 1
        st.rerun()
    else:
        # Gesture can only provide location, so retry
        st.warning(
            f"Missing: {', '.join(missing)}. Please say the full command again."
        )


def _live_multimodal_gesture(col_left, col_right) -> None:
    """Handle fallback gesture step for multimodal input.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """
    voice_cmd: RobotCommand = st.session_state.live_voice_cmd
    action_label = (
        voice_cmd.action.value if voice_cmd and voice_cmd.action else "unknown"
    )
    missing = validate_command(voice_cmd)

    with col_left:
        st.caption(
            f"Step 2 — Voice: {action_label}. Gesture the missing: {', '.join(missing)}"
        )

    if config.GESTURE_INPUT == "webcam":
        _live_multimodal_gesture_webcam(voice_cmd, missing, col_left, col_right)
    else:
        _live_multimodal_gesture_buttons(voice_cmd, col_left, col_right)


def _live_multimodal_gesture_buttons(
    voice_cmd: RobotCommand, col_left, col_right
) -> None:
    """
    Handle button gesture fallback.

    Args:
        voice_cmd: Voice command collected.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        with st.form("live_mm_gesture_form"):
            choice = st.radio(
                "Location",
                options=list(LOCATION_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            confirmed = st.form_submit_button(
                "Confirm Location", type="primary"
            )

    if confirmed:
        location = LOCATION_OPTIONS[choice]
        gesture_ts = time.time()

        gesture_cmd = RobotCommand(
            mode=Mode.gesture,
            location=location,
            confidence=1.0,
            location_confidence=1.0,
        )
        fusion_result = fuse_inputs(
            voice_cmd=voice_cmd,
            gesture_cmd=gesture_cmd,
            voice_ts=st.session_state.live_voice_ts,
            gesture_ts=gesture_ts,
        )

        final_cmd = fusion_result.command
        final_cmd.timestamp = _iso_now()
        st.session_state.live_cmd = final_cmd
        st.rerun()


def _live_multimodal_gesture_webcam(
    voice_cmd: RobotCommand, missing: list[str], col_left, col_right
) -> None:
    """
    Handle webcam gesture fallback.

    Args:
        voice_cmd: Voice command collected.
        missing: Missing fields after voice parsing.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    try:
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        with col_left:
            st.error(
                "streamlit-webrtc is not installed. Run: pip install streamlit-webrtc"
            )
        return

    with col_left:
        overlay_on = st.checkbox(
            "Show hand overlay",
            value=config.GESTURE_OVERLAY_ENABLED,
            key="live_mm_overlay",
        )
        GestureProcessor = _get_gesture_processor_class(
            overlay_enabled=overlay_on
        )

        if "location" in missing:
            st.caption(
                f"Hold hand left or right of centre for {config.GESTURE_STABILITY_SECS}s to auto-capture — or click Confirm."
            )

        ctx = webrtc_streamer(
            key="live_mm_gesture",
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    if ctx.video_processor:
        ctx.video_processor.overlay_enabled = overlay_on

        with col_left:
            confirm_clicked = st.button(
                "Confirm Location", key="live_mm_location_confirm"
            )

        status_placeholder = col_right.empty()

        if confirm_clicked:
            with ctx.video_processor.lock:
                inferred_loc = ctx.video_processor.inferred_location
                loc_conf_val = ctx.video_processor.inferred_location_confidence
                gesture_result = ctx.video_processor.gesture_result
            if inferred_loc is None:
                with col_left:
                    st.warning(
                        "No hand detected. Show your hand to the camera."
                    )
            else:
                gesture_cmd = RobotCommand(
                    mode=Mode.gesture,
                    location=inferred_loc,
                    confidence=(
                        gesture_result.confidence if gesture_result else 1.0
                    ),
                    location_confidence=loc_conf_val,
                )
                gesture_ts = time.time()
                fusion_result = fuse_inputs(
                    voice_cmd=voice_cmd,
                    gesture_cmd=gesture_cmd,
                    voice_ts=st.session_state.live_voice_ts,
                    gesture_ts=gesture_ts,
                )
                final_cmd = fusion_result.command
                final_cmd.timestamp = _iso_now()
                st.session_state.live_cmd = final_cmd
                st.rerun()

        # Show live inferred location during the fallback step
        while ctx.state.playing:
            with ctx.video_processor.lock:
                inferred_loc = ctx.video_processor.inferred_location

            with status_placeholder.container():
                st.metric(
                    "Location", inferred_loc.value if inferred_loc else "-"
                )
            time.sleep(0.1)


def _run_voice_input(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle trial voice input.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    if config.VOICE_INPUT == "mic":
        _trial_voice_mic(trial, col_left, col_right)
    else:
        _trial_voice_typed(trial, col_left, col_right)


def _trial_voice_typed(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle typed voice input for trial mode.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        st.write("**Type your command below:**")
        st.caption(
            "Use natural language, e.g. 'pick up the red cube and put it on the left'."
        )

        with st.form("voice_form"):
            text = st.text_input(
                "Command",
                placeholder="e.g. pick the red cube left",
            )
            submitted = st.form_submit_button("Submit Command", type="primary")

    if submitted and text.strip():
        cmd = parse_text_to_intent(text.strip())
        cmd.latency_ms = _elapsed_ms(st.session_state.trial_start_s)
        cmd.timestamp = _iso_now()
        _preview_result(cmd=cmd, voice_ts=_iso_now())
    elif submitted:
        with col_left:
            st.warning("Please enter a command before submitting.")


def _trial_voice_mic(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle microphone voice input for trial mode.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        st.write("**Record your command:**")
        st.caption("Click the microphone to start recording, speak, then stop.")
        audio_bytes = st.audio_input(
            "Record a voice command", key="trial_audio"
        )

    if audio_bytes is not None:
        with col_left:
            with st.spinner("Transcribing…"):
                from voice.speech import transcribe_audio_bytes

                text = transcribe_audio_bytes(audio_bytes.getvalue())

            if text:
                st.write(f"**Heard:** {text}")
                cmd = parse_text_to_intent(text)
                cmd.latency_ms = _elapsed_ms(st.session_state.trial_start_s)
                cmd.timestamp = _iso_now()
                _preview_result(cmd=cmd, voice_ts=_iso_now())
            else:
                st.warning("No speech detected. Please try again.")


def _run_gesture_input(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle trial gesture input flow.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    step = st.session_state.gesture_step
    if step == 0:
        _gesture_step_action(trial, col_left, col_right)
    else:
        _gesture_step_location(trial, col_left, col_right)


def _gesture_step_action(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle action step of gesture input.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        st.write("**Step 1 of 2 — Select an action gesture:**")
        st.caption(
            "✊ Closed Fist = pick  |  🤚 Open Palm = stop  |  ✌️ Victory = place"
        )

    if config.GESTURE_INPUT == "webcam":
        _trial_gesture_webcam_action(trial, col_left, col_right)
    else:
        _trial_gesture_buttons_action(trial, col_left, col_right)


def _trial_gesture_buttons_action(
    trial: TrialDefinition, col_left, col_right
) -> None:
    """
    Handle button action gesture input.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """
    with col_left:
        with st.form("gesture_action_form"):
            choice = st.radio(
                "Action gesture",
                options=list(ACTION_GESTURE_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            confirmed = st.form_submit_button(
                "Confirm Action Gesture", type="primary"
            )

    if confirmed:
        gesture_name = ACTION_GESTURE_OPTIONS[choice]
        intent = map_gesture_to_intent(gesture_name)
        action: Action = intent["action"]
        st.session_state.gesture_action = action

        if action in (Action.stop, Action.cancel):
            cmd = RobotCommand(
                mode=Mode.gesture,
                action=action,
                object=trial.expected_object,
                location=None,
                confidence=1.0,
                latency_ms=_elapsed_ms(st.session_state.trial_start_s),
                timestamp=_iso_now(),
            )
            _preview_result(cmd=cmd)
        else:
            st.session_state.gesture_step = 1
            st.rerun()


def _trial_gesture_webcam_action(
    trial: TrialDefinition, col_left, col_right
) -> None:
    """
    Handle webcam action gesture input.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """
    try:
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        with col_left:
            st.error(
                "streamlit-webrtc is not installed. Run: pip install streamlit-webrtc"
            )
        return

    GestureProcessor = _get_gesture_processor_class(
        overlay_enabled=config.GESTURE_OVERLAY_ENABLED
    )

    with col_left:
        st.caption(
            f"Hold your gesture steady for {config.GESTURE_STABILITY_SECS}s to auto-capture — or click Confirm Gesture."
        )
        ctx = webrtc_streamer(
            key="trial_gesture_action",
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            translations={"stop": "Confirm Gesture"},
        )

    # Confirm gesture capture the latest cached gesture state
    if not ctx.state.playing:
        cached = st.session_state.pop("_trial_gesture_action_cache", None)
        if cached is not None:
            if cached.gesture_label not in _ACTION_GESTURES:
                with col_left:
                    st.warning(
                        "No action gesture detected. Show ✊ Closed Fist, 🤚 Open Palm, or ✌️ Victory."
                    )
            else:
                intent = map_gesture_to_intent(cached.gesture_label)
                action = intent["action"]
                st.session_state.gesture_action = action
                if action in (Action.stop, Action.cancel):
                    cmd = RobotCommand(
                        mode=Mode.gesture,
                        action=action,
                        object=trial.expected_object,
                        location=None,
                        confidence=cached.confidence,
                        latency_ms=_elapsed_ms(st.session_state.trial_start_s),
                        timestamp=_iso_now(),
                    )
                    _preview_result(cmd=cmd)
                else:
                    st.session_state.gesture_step = 1
                    st.rerun()

    if ctx.video_processor:
        status_placeholder = col_right.empty()

        stable_label: Optional[str] = None
        stable_since: Optional[float] = None
        while ctx.state.playing:
            with ctx.video_processor.lock:
                gesture_result = ctx.video_processor.gesture_result
                last_action = ctx.video_processor.last_action_result
                thumb_up = ctx.video_processor.thumb_up_confirm

            # Cache current gesture for confirmation
            st.session_state["_trial_gesture_action_cache"] = (
                last_action if (thumb_up and last_action) else gesture_result
            )

            current_label = (
                gesture_result.gesture_label
                if gesture_result
                and gesture_result.gesture_label in _ACTION_GESTURES
                else None
            )
            if current_label != stable_label:
                stable_label = current_label
                stable_since = time.time() if current_label else None

            if (
                stable_label
                and stable_since
                and (time.time() - stable_since)
                >= config.GESTURE_STABILITY_SECS
            ):
                intent = map_gesture_to_intent(stable_label)
                action = intent["action"]
                st.session_state.gesture_action = action

                if action in (Action.stop, Action.cancel):
                    cmd = RobotCommand(
                        mode=Mode.gesture,
                        action=action,
                        object=trial.expected_object,
                        location=None,
                        confidence=(
                            gesture_result.confidence if gesture_result else 1.0
                        ),
                        latency_ms=_elapsed_ms(st.session_state.trial_start_s),
                        timestamp=_iso_now(),
                    )
                    _preview_result(cmd=cmd)
                else:
                    st.session_state.gesture_step = 1
                    st.rerun()

            with status_placeholder.container():
                _render_gesture_status(gesture_result)
                if stable_label and stable_since:
                    elapsed = time.time() - stable_since
                    st.progress(
                        min(elapsed / config.GESTURE_STABILITY_SECS, 1.0),
                        text=f"Holding {stable_label}… {elapsed:.1f}s",
                    )
                if (
                    gesture_result
                    and gesture_result.gesture_label == CONFIRM_GESTURE
                    and last_action
                ):
                    st.info(
                        f"Thumb_Up — will confirm: **{last_action.gesture_label}**"
                    )
            time.sleep(0.1)


def _gesture_step_location(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle location step of gesture input.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    action: Action = st.session_state.gesture_action

    with col_left:
        st.write("**Step 2 of 2 — Indicate the target location:**")
        st.caption(
            f"Action captured: **{action.value}** — now position your hand or select below."
        )

    if config.GESTURE_INPUT == "webcam":
        _trial_gesture_webcam_location(trial, action, col_left, col_right)
    else:
        _trial_gesture_buttons_location(trial, action, col_left, col_right)


def _trial_gesture_buttons_location(
    trial: TrialDefinition, action: Action, col_left, col_right
) -> None:
    """
    Handle button location selection.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        with st.form("gesture_location_form"):
            choice = st.radio(
                "Location",
                options=list(LOCATION_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            confirmed = st.form_submit_button(
                "Confirm Location", type="primary"
            )

    if confirmed:
        location = LOCATION_OPTIONS[choice]
        cmd = RobotCommand(
            mode=Mode.gesture,
            action=action,
            object=trial.expected_object,
            location=location,
            confidence=1.0,
            latency_ms=_elapsed_ms(st.session_state.trial_start_s),
            timestamp=_iso_now(),
        )
        _preview_result(cmd=cmd)


def _trial_gesture_webcam_location(
    trial: TrialDefinition, action: Action, col_left, col_right
) -> None:
    """
    Handle webcom location selection.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """
    try:
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        with col_left:
            st.error(
                "streamlit-webrtc is not installed. Run: pip install streamlit-webrtc"
            )
        return

    GestureProcessor = _get_gesture_processor_class(
        overlay_enabled=config.GESTURE_OVERLAY_ENABLED
    )

    with col_left:
        st.caption(
            "Position your hand left or right of centre, then click Confirm Gesture."
        )
        ctx = webrtc_streamer(
            key="trial_gesture_location",
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            translations={"stop": "Confirm Gesture"},
        )

    # Confirm Gesture captures the latest cached location state
    if not ctx.state.playing:
        cached = st.session_state.pop("_trial_gesture_loc_cache", None)
        if cached is not None:
            inferred_loc, confidence = cached
            if inferred_loc is not None:
                cmd = RobotCommand(
                    mode=Mode.gesture,
                    action=action,
                    object=trial.expected_object,
                    location=inferred_loc,
                    confidence=confidence,
                    latency_ms=_elapsed_ms(st.session_state.trial_start_s),
                    timestamp=_iso_now(),
                )
                _preview_result(cmd=cmd)

    if ctx.video_processor:

        status_placeholder = col_right.empty()

        while ctx.state.playing:
            with ctx.video_processor.lock:
                inferred_loc = ctx.video_processor.inferred_location
                gesture_result = ctx.video_processor.gesture_result

            # Cache current location for confirmation
            st.session_state["_trial_gesture_loc_cache"] = (
                inferred_loc,
                gesture_result.confidence if gesture_result else 1.0,
            )

            with status_placeholder.container():
                _render_gesture_status(
                    gesture_result,
                    show_location=True,
                    inferred_loc=inferred_loc,
                )
            time.sleep(0.1)


def _run_multimodal_input(trial: TrialDefinition, col_left, col_right) -> None:
    """
    Handle trial multimodal input flow.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column
    """

    step = st.session_state.multimodal_step
    if step == 0:
        _multimodal_step_voice(col_left, col_right)
    else:
        _multimodal_step_gesture(trial, col_left, col_right)


def _multimodal_step_voice(col_left, col_right) -> None:
    """
    Handle simultaneous voice and gesture input for trial mode.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    # Start of active voice window for simultaneous fusion
    if st.session_state.voice_step_start_ts is None:
        st.session_state.voice_step_start_ts = time.time()

    with col_left:
        st.write(
            "**Speak your command — gesture simultaneously to add location:**"
        )

    mm_ctx = None
    if config.GESTURE_INPUT == "webcam":
        try:
            from streamlit_webrtc import webrtc_streamer

            GestureProcessor = _get_gesture_processor_class(
                overlay_enabled=config.GESTURE_OVERLAY_ENABLED
            )
            with col_left:
                mm_ctx = webrtc_streamer(
                    key="trial_mm_gesture",
                    video_processor_factory=GestureProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                )

            # Show current gesture state during voice input
            if mm_ctx.video_processor:
                with mm_ctx.video_processor.lock:
                    _cur_result = mm_ctx.video_processor.gesture_result
                    _cur_loc = mm_ctx.video_processor.inferred_location
                with col_right:
                    _render_gesture_status(
                        _cur_result, show_location=True, inferred_loc=_cur_loc
                    )
        except ImportError:
            pass

    if config.VOICE_INPUT == "mic":
        _multimodal_voice_mic(mm_ctx, col_left, col_right)
    else:
        _multimodal_voice_typed(mm_ctx, col_left, col_right)


def _multimodal_voice_typed(mm_ctx, col_left, col_right) -> None:
    """
    Handle typed voice input for multimodal trial mode.

    Args:
        mm_ctx: Webcam context for simultaneous gesture capture.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        st.caption("Position your hand in frame, then type your command.")
        with st.form("multimodal_voice_form"):
            text = st.text_input(
                "Voice command",
                placeholder="e.g. pick up the red cube",
            )
            submitted = st.form_submit_button("Submit Voice", type="primary")

    if submitted and text.strip():
        pre_loc, gesture_det_ts, gesture_result, loc_conf = (
            _read_gesture_state_for_fusion(
                mm_ctx, st.session_state.voice_step_start_ts
            )
        )
        _multimodal_process_voice_text(
            text.strip(),
            pre_loc=pre_loc,
            gesture_ts=gesture_det_ts,
            gesture_confidence=(
                gesture_result.confidence if gesture_result else 1.0
            ),
            location_confidence=loc_conf,
        )
    elif submitted:
        with col_left:
            st.warning("Please enter a command before submitting.")


def _multimodal_voice_mic(mm_ctx, col_left, col_right) -> None:
    """
    Handle microphone voice input for multimodal trial mode.

    Args:
        mm_ctx: Webcam context for simultaneous gesture capture.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        st.caption("Position your hand in frame, then record your command.")
        audio_bytes = st.audio_input(
            "Record a voice command", key="trial_mm_audio"
        )

    if audio_bytes is not None:
        # Capture gesture state before transcription latency
        pre_loc, gesture_det_ts, gesture_result, loc_conf = (
            _read_gesture_state_for_fusion(
                mm_ctx, st.session_state.voice_step_start_ts
            )
        )

        with col_left:
            with st.spinner("Transcribing…"):
                from voice.speech import transcribe_audio_bytes

                text = transcribe_audio_bytes(audio_bytes.getvalue())

            if text:
                st.write(f"**Heard:** {text}")
                _multimodal_process_voice_text(
                    text,
                    pre_loc=pre_loc,
                    gesture_ts=gesture_det_ts,
                    gesture_confidence=(
                        gesture_result.confidence if gesture_result else 1.0
                    ),
                    location_confidence=loc_conf,
                )
            else:
                st.warning("No speech detected. Please try again.")


def _multimodal_process_voice_text(
    text: str,
    pre_loc: Optional[Location] = None,
    gesture_ts: Optional[float] = None,
    gesture_confidence: float = 1.0,
    location_confidence: float = 1.0,
) -> None:
    """
    Parse voice input and apply multimodal fusion.

    Args:
        text: Transcribed or typed voice input.
        pre_loc: Gesture-derived location, if available.
        gesture_ts: Gesture timestamp.
        gesture_confidence: Gesture confidence score.
        location_confidence: Confidence of inferred location.
    """

    voice_cmd = parse_text_to_intent(text)
    voice_ts = time.time()

    st.session_state.voice_cmd = voice_cmd
    st.session_state.voice_ts = voice_ts

    missing = validate_command(voice_cmd)

    if voice_cmd.action in (Action.stop, Action.cancel) or not missing:
        # Command is complete, no gesture needed
        voice_cmd.mode = Mode.multimodal
        voice_cmd.latency_ms = _elapsed_ms(st.session_state.trial_start_s)
        voice_cmd.timestamp = _iso_now()
        _preview_result(cmd=voice_cmd, voice_ts=_ts_iso(voice_ts))
    elif voice_cmd.action is None:
        st.warning(
            "No action recognised in that command. "
            "Try again, e.g. 'pick up the red cube'."
        )
    elif "location" in missing and pre_loc is not None:
        # Fuse immediately when location was capture during the voice window
        gesture_cmd = RobotCommand(
            mode=Mode.gesture,
            location=pre_loc,
            confidence=gesture_confidence,
            location_confidence=location_confidence,
        )

        g_ts = gesture_ts if gesture_ts is not None else voice_ts
        fusion_result = fuse_inputs(
            voice_cmd=voice_cmd,
            gesture_cmd=gesture_cmd,
            voice_ts=voice_ts,
            gesture_ts=g_ts,
        )

        final_cmd = fusion_result.command
        final_cmd.latency_ms = _elapsed_ms(st.session_state.trial_start_s)
        final_cmd.timestamp = _iso_now()

        _preview_result(
            cmd=final_cmd,
            fusion=fusion_result,
            conflict_flag=bool(fusion_result.conflict_fields),
            voice_ts=_ts_iso(voice_ts),
            gesture_ts=_ts_iso(g_ts),
            fusion_within_window=fusion_result.within_window,
        )

    elif missing == ["location"]:
        st.info("Location missing — please gesture left or right.")
        st.session_state.multimodal_step = 1
        st.rerun()

    else:
        st.warning(
            f"Missing: {', '.join(missing)}. Please say the full command again."
        )


def _multimodal_step_gesture(
    trial: TrialDefinition, col_left, col_right
) -> None:
    """
    Handle gesture fallback step for multimodal trial mode.

    Args:
        trial: Current trial definition.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    voice_cmd: RobotCommand = st.session_state.voice_cmd
    action_label = (
        voice_cmd.action.value if voice_cmd and voice_cmd.action else "unknown"
    )

    with col_left:
        st.write("**Step 2 — Gesture the target location:**")
        st.caption(
            f"Voice captured: **{action_label}** — now indicate the location."
        )

    if config.GESTURE_INPUT == "webcam":
        _multimodal_gesture_webcam(voice_cmd, col_left, col_right)
    else:
        _multimodal_gesture_buttons(voice_cmd, col_left, col_right)


def _multimodal_gesture_buttons(
    voice_cmd: RobotCommand, col_left, col_right
) -> None:
    """
    Handle button gesture fallback.

    Args:
        voice_cmd: Voice command collected.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    with col_left:
        with st.form("multimodal_gesture_form"):
            choice = st.radio(
                "Location",
                options=list(LOCATION_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            confirmed = st.form_submit_button(
                "Confirm Location", type="primary"
            )

    if confirmed:
        location = LOCATION_OPTIONS[choice]
        gesture_ts = time.time()

        gesture_cmd = RobotCommand(
            mode=Mode.gesture,
            location=location,
            confidence=1.0,
            location_confidence=1.0,
        )

        fusion_result: FusionResult = fuse_inputs(
            voice_cmd=st.session_state.voice_cmd,
            gesture_cmd=gesture_cmd,
            voice_ts=st.session_state.voice_ts,
            gesture_ts=gesture_ts,
        )

        final_cmd = fusion_result.command
        final_cmd.latency_ms = _elapsed_ms(st.session_state.trial_start_s)
        final_cmd.timestamp = _iso_now()

        _preview_result(
            cmd=final_cmd,
            fusion=fusion_result,
            conflict_flag=bool(fusion_result.conflict_fields),
            voice_ts=_ts_iso(st.session_state.voice_ts),
            gesture_ts=_ts_iso(gesture_ts),
            fusion_within_window=fusion_result.within_window,
        )


def _multimodal_gesture_webcam(
    voice_cmd: RobotCommand, col_left, col_right
) -> None:
    """
    Handle webcam gesture fallback.

    Args:
        voice_cmd: Voice command collected.
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    try:
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        with col_left:
            st.error(
                "streamlit-webrtc is not installed. Run: pip install streamlit-webrtc"
            )
        return

    GestureProcessor = _get_gesture_processor_class(
        overlay_enabled=config.GESTURE_OVERLAY_ENABLED
    )

    with col_left:
        st.caption(
            f"Hold hand left or right of centre for {config.GESTURE_STABILITY_SECS}s to auto-capture — or click Confirm."
        )
        ctx = webrtc_streamer(
            key="trial_mm_gesture",
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    if ctx.video_processor:
        with col_left:
            confirm_clicked = st.button(
                "Confirm Location", key="trial_mm_location_confirm"
            )

        status_placeholder = col_right.empty()

        if confirm_clicked:
            with ctx.video_processor.lock:
                inferred_loc = ctx.video_processor.inferred_location
                loc_conf_val = ctx.video_processor.inferred_location_confidence
                gesture_result = ctx.video_processor.gesture_result

            if inferred_loc is None:
                with col_left:
                    st.warning(
                        "No hand detected. Show your hand to the camera."
                    )
            else:
                gesture_cmd = RobotCommand(
                    mode=Mode.gesture,
                    location=inferred_loc,
                    confidence=(
                        gesture_result.confidence if gesture_result else 1.0
                    ),
                    location_confidence=loc_conf_val,
                )
                gesture_ts = time.time()

                fusion_result = fuse_inputs(
                    voice_cmd=voice_cmd,
                    gesture_cmd=gesture_cmd,
                    voice_ts=st.session_state.voice_ts,
                    gesture_ts=gesture_ts,
                )

                final_cmd = fusion_result.command
                final_cmd.latency_ms = _elapsed_ms(
                    st.session_state.trial_start_s
                )
                final_cmd.timestamp = _iso_now()

                _preview_result(
                    cmd=final_cmd,
                    fusion=fusion_result,
                    conflict_flag=bool(fusion_result.conflict_fields),
                    voice_ts=_ts_iso(st.session_state.voice_ts),
                    gesture_ts=_ts_iso(gesture_ts),
                    fusion_within_window=fusion_result.within_window,
                )

        while ctx.state.playing:
            with ctx.video_processor.lock:
                inferred_loc = ctx.video_processor.inferred_location

            with status_placeholder.container():
                st.metric(
                    "Location", inferred_loc.value if inferred_loc else "—"
                )
            time.sleep(0.1)


def _preview_result(
    cmd: RobotCommand,
    fusion: Optional[FusionResult] = None,
    conflict_flag: bool = False,
    voice_ts: Optional[str] = None,
    gesture_ts: Optional[str] = None,
    fusion_within_window: Optional[bool] = None,
) -> None:
    """
    Store trial result data and move to the result page.

    Args:
        cmd: Parsed or fused command.
        fusion: Fusion metadata, if available.
        conflict_flag: Whether fusion conflict occurred.
        voice_ts: Voice timestamp string.
        gesture_ts: Gesture timestamp string.
        fusion_within_window: Whether fusion occurred within the time window.
    """

    st.session_state.pending_cmd = cmd
    st.session_state.pending_fusion = fusion
    st.session_state.pending_conflict_flag = conflict_flag
    st.session_state.pending_voice_ts = voice_ts
    st.session_state.pending_gesture_ts = gesture_ts
    st.session_state.pending_fusion_within_window = fusion_within_window
    st.session_state.phase = "trial_result"
    st.rerun()


def _run_result_page(col_left, col_right) -> None:
    """
    Display recognised command and trial controls.

    Args:
        col_left: Left Streamlit column.
        col_right: Right Streamlit column.
    """

    cmd: RobotCommand = st.session_state.pending_cmd
    fusion: Optional[FusionResult] = st.session_state.pending_fusion
    runner: ExperimentRunner = st.session_state.runner
    trial = runner.get_current_trial()

    correct = (
        _is_correct(trial, cmd.action, cmd.object, cmd.location)
        if trial
        else None
    )

    # Command output and correctness
    with col_right:
        render_command_panel(cmd, correct=correct)
        st.caption(f"Latency: {cmd.latency_ms:.0f} ms")

        if fusion and fusion.conflict_fields:
            winner = fusion.field_source.get(fusion.conflict_fields[0], "voice")
            st.warning(
                f"Conflict in: {', '.join(fusion.conflict_fields)} — {winner} values used."
            )

        if fusion and not fusion.within_window:
            st.warning(
                "Inputs were outside the fusion time window — command not fused."
            )

        if fusion and fusion.needs_confirmation:
            st.warning(f"Ambiguous: {fusion.ambiguity_reason}")

        if fusion and fusion.field_source:
            with st.expander("Fusion details"):
                prov_str = ", ".join(
                    f"{k}: {v}" for k, v in fusion.field_source.items()
                )
                st.markdown(f"**Provenance:** {prov_str}")

                diag = fusion.diagnostics
                if diag.get("temporal_score") is not None:
                    st.markdown(
                        f"**Temporal:** gap={diag.get('temporal_gap', 0):.2f}s, "
                        f"score={diag['temporal_score']:.2f}"
                    )
                if diag.get("confidence_decision_reasons"):
                    for field, reason in diag[
                        "confidence_decision_reasons"
                    ].items():
                        st.caption(f"{field}: {reason}")

        if st.session_state.correction_count > 0:
            st.caption(
                f"Corrections so far: {st.session_state.correction_count}"
            )

    # Accept / Retry controls
    with col_left:
        st.markdown(
            "**Command recognised — review the result, then accept or retry.**"
        )
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("Accept & Continue ->", type="primary"):
                _submit_and_advance()

        with btn_col2:
            if st.button("Retry (count as correction)"):
                _retry_trial()


def _submit_and_advance() -> None:
    """
    Submit accepted result and advance to the next trial.
    """

    runner: ExperimentRunner = st.session_state.runner
    logger: SessionLogger = st.session_state.logger
    cmd: RobotCommand = st.session_state.pending_cmd

    result = runner.submit_result(
        predicted_action=cmd.action,
        predicted_object=cmd.object,
        predicted_location=cmd.location,
        latency_ms=cmd.latency_ms,
        correction_count=st.session_state.correction_count,
        conflict_flag=st.session_state.pending_conflict_flag,
        voice_timestamp=st.session_state.pending_voice_ts,
        gesture_timestamp=st.session_state.pending_gesture_ts,
        fusion_within_window=st.session_state.pending_fusion_within_window,
        confidence=cmd.confidence,
    )
    logger.log_trial(result)
    runner.advance()

    if runner.get_current_trial() is None:
        st.session_state.phase = "done"
    else:
        st.session_state.phase = "trial_input"

    _reset_trial_state()
    st.rerun()


def _retry_trial() -> None:
    """
    Reset the current trial and increment correction count.
    """

    st.session_state.correction_count += 1
    st.session_state.phase = "trial_input"
    st.session_state.trial_start_s = time.time()
    st.session_state.gesture_step = 0
    st.session_state.gesture_action = None
    st.session_state.multimodal_step = 0
    st.session_state.voice_cmd = None
    st.session_state.voice_ts = None
    st.session_state.voice_step_start_ts = None
    st.session_state.pop("_trial_gesture_action_cache", None)
    st.session_state.pop("_trial_gesture_loc_cache", None)
    st.rerun()


def _reset_trial_state() -> None:
    """
    Clear per-trial state for the next trial.
    """

    st.session_state.trial_start_s = time.time()
    st.session_state.gesture_step = 0
    st.session_state.gesture_action = None
    st.session_state.multimodal_step = 0
    st.session_state.voice_cmd = None
    st.session_state.voice_ts = None
    st.session_state.voice_step_start_ts = None
    st.session_state.correction_count = 0
    st.session_state.pending_cmd = None
    st.session_state.pending_fusion = None
    st.session_state.pending_conflict_flag = False
    st.session_state.pending_voice_ts = None
    st.session_state.pending_gesture_ts = None
    st.session_state.pending_fusion_within_window = None
    st.session_state.pop("_trial_gesture_action_cache", None)
    st.session_state.pop("_trial_gesture_loc_cache", None)


def _run_done_page() -> None:
    """
    Display session summary and log location.
    """

    runner: Optional[ExperimentRunner] = st.session_state.runner
    logger: Optional[SessionLogger] = st.session_state.logger

    if runner is None or logger is None:
        st.error("Session state lost. Please refresh and start a new session.")
        return

    summary = runner.get_summary()
    render_session_summary(summary)

    csv_path = logger.save()
    st.markdown("---")
    st.success(f"Session log saved to: `{csv_path}`")

    if st.button("Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def _page_home() -> None:
    """
    Render the home page.
    """

    st.logo("🤖")
    st.title("🤖 Multimodal HRI Command Hub")
    st.caption("Voice and gesture input for structured robot commands.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Live Interaction")
        st.write(
            "Freeform voice and gesture commands in real time. "
            "Select an input mode and start interacting — no prompts, no logging."
        )
        if st.button(
            "Start Live Mode", type="primary", use_container_width=True
        ):
            st.switch_page(st.session_state._live_page_ref)

    with col2:
        st.markdown("### Experiment")
        st.write(
            "Structured trials with task prompts, correctness checking, "
            "and session logging."
        )
        participant_id = st.text_input(
            "Participant ID",
            value="P01",
            placeholder="e.g. P01",
            key="home_participant_id",
            help="Leave blank to use system evaluation mode.",
        )
        if st.button("Start Experiment", use_container_width=True):
            pid = participant_id.strip() or "system"
            st.session_state.runner = ExperimentRunner(participant_id=pid)
            st.session_state.logger = SessionLogger(participant_id=pid)
            st.session_state.participant_id = pid
            st.session_state.phase = "trial_input"
            st.session_state.trial_start_s = time.time()
            st.switch_page(st.session_state._trial_page_ref)


_LIVE_DASHBOARD_CSS = """
<style>
section.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0.25rem !important;
}

div[data-testid="stVerticalBlock"] > div {
    gap: 0.45rem;
}

video {
    max-height: 165px !important;
    object-fit: contain;
}

div[data-testid="stAudioInput"] {
    margin-top: 0.15rem !important;
    margin-bottom: 0.35rem !important;
}

div[data-testid="stButton"] {
    margin-top: 0.1rem !important;
    margin-bottom: 0.25rem !important;
}

p {
    margin-bottom: 0.35rem !important;
}

[data-testid="stHorizontalBlock"]:has(.live-right-panel) {
    align-items: flex-start !important;
}

[data-testid="column"]:has(.live-right-panel),
[data-testid="stColumn"]:has(.live-right-panel) {
    position: sticky !important;
    top: 3.5rem !important;
    align-self: flex-start !important;
    height: fit-content !important;
}
</style>
"""


def _page_live() -> None:
    """
    Render the live interaction page.
    """

    st.logo("🤖")
    st.markdown(_LIVE_DASHBOARD_CSS, unsafe_allow_html=True)
    _run_live_page()


def _page_trial() -> None:
    """
    Render the trial mode page.
    """

    st.logo("🤖")

    if st.session_state.runner is None:
        _run_trial_setup()
        return

    runner: ExperimentRunner = st.session_state.runner
    current_index, total = runner.progress()
    trial = runner.get_current_trial()
    phase: str = st.session_state.phase

    if phase == "done" or trial is None:
        _run_done_page()
        return

    render_mode_badge(trial.condition)
    render_progress(current_index, total)
    render_trial_prompt(trial)

    col_left, col_right = st.columns([3, 2])

    if phase == "trial_input":
        if trial.condition == Mode.voice:
            _run_voice_input(trial, col_left, col_right)
        elif trial.condition == Mode.gesture:
            _run_gesture_input(trial, col_left, col_right)
        elif trial.condition == Mode.multimodal:
            _run_multimodal_input(trial, col_left, col_right)

    elif phase == "trial_result":
        _run_result_page(col_left, col_right)


def run_app() -> None:
    """
    Configure and run the Streamlit application.
    """

    st.set_page_config(
        page_title="Multimodal HRI Command Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_state()

    live_page = st.Page(_page_live, title="Live")
    trial_page = st.Page(_page_trial, title="Trial")
    st.session_state._live_page_ref = live_page
    st.session_state._trial_page_ref = trial_page

    pages = [
        st.Page(_page_home, title="Home"),
        live_page,
        trial_page,
    ]
    pg = st.navigation(pages, position="top")
    pg.run()


def _elapsed_ms(start_s: float) -> float:
    """
    Helper method to return elapsed time in milliseconds.

    Args:
        start_s: Start time in seconds.

    Returns:
        Elapsed time in milliseconds.
    """

    return round((time.time() - start_s) * 1000, 2)


def _iso_now() -> str:
    """
    Helper method to return current UTC time as an ISO 8601 string.

    Returns:
        Current UTC timestamp string.
    """

    return datetime.now(timezone.utc).isoformat()


def _ts_iso(unix_ts: float) -> str:
    """
    Helper method to convert Unix timestamp to ISO 8601 format.

    Args:
        unix_ts: Unix timestamp in seconds.

    Returns:
        ISO 8601 timestamp string.
    """

    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
