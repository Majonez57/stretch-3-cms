# Multimodal Fusion Design

## Overview

The system implements **decision-level multimodal fusion**, combining structured outputs from voice and gesture modalities into a single unified robot command.

Unlike early fusion approaches that operate on raw sensor data, this system performs fusion at the **semantic level**, merging command fields that have been interpreted already:

- `action`
- `object`
- `location`

The fusion process integrates:
- Temporal alignment
- Confidence-based decision making
- Conflict resolution
- Missing field recovery

## Fusion Inputs

Each modality produces a partial `RobotCommand`:

### Voice Input
- Extracted from natural language
- Provides: `action`, `object`, `location`
- Includes per-field confidence scores

### Gesture Input
- Derived from hand pose and spatial position
- Provides:
  - `action` (gesture classification)
  - `location` (left/right inference)
- Includes confidence scores for gesture and location

These inputs are fused into a single command using the algorithm described below.


## Fusion Pipeline
The fusion pipeline performs decision-level fusion of voice and gesture commands. Inputs are first temporally aligned within a fusion window. Field-level fusion is then applied independently to action, object, and location using agreement, conflict resolution, and ambiguity handling. The final confidence score is computed from merged field confidences and adjusted using temporal alignment.

![Fusion Pipeline Diagram](/assets/fusion-pipeline-diagram.png)


## Temporal Alignment

Fusion is gated by a temporal constraint to ensure inputs are related.

- A maximum window of **3 seconds** is used: `FUSION_WINDOW_SECONDS = 3.0`

The temporal score is defined as:

$$
\text{score} = 1 - \left(\frac{\text{gap}}{\text{window}}\right)^{\alpha}
$$

Where:
- $\text{gap}$ is the time difference between voice and gesture inputs
- $\text{window}$ is the maximum fusion window
- $\alpha$ is the temporal decay exponent

If $\text{gap} > \text{window}$,  fusion is rejected .

Otherwise, the temporal score is used to scale the final confidence.


## Field-Level Fusion

Each field (`action`, `object`, `location`) is processed independently.

### Case 1: Single-Modality Contribution

If only one modality provides a value:

- Accept if confidence exceeds threshold:
  - Voice: `>= 0.4`
  - Gesture:
    - Action: `>= 0.5`
    - Location: `>= 0.5`

Otherwise, field is discarded.

---

### Case 2: Agreement

If both modalities agree:

- Value is accepted
- Confidence is taken as: $\max(\text{voice}, \text{gesture})$
- Marked as `"agreement"`

Agreement contributes positively to final confidence through the agreement bonus.

---

### Case 3: Conflict Resolution

If modalities disagree:

- Compare confidence scores
- Apply **voice bias margin**:
    ```
    if voice_conf >= gesture_conf - margin:
        choose voice
    else:
        choose gesture
    ```
    Where: `FUSION_VOICE_BIAS_MARGIN = 0.15`

This reflects the assumption that:
- Voice is generally more precise for semantics
- Gesture provides spatial context

---

### Ambiguity Detection

If confidence gap is small: 

$$
|\text{voice}_{conf} - \text{gesture}_{conf}| < 0.15
$$

Then the result is flagged as **ambiguous** and may require confirmation.

---

## Handling Missing Information

Fusion supports **complementary input**:

- Voice may omit location
- Gesture provides location
- Combined result becomes complete

If fields remain missing:
- Penalty is applied to final confidence


## Confidence Computation

Final confidence is computed as:

$$
\text{confidence} =
(\text{base} + \text{adjustments}) \times \text{temporal score}
$$

### Base Confidence
Average of merged field confidences.

### Adjustments
- Agreement bonus: `+0.1` per agreeing field
- Conflict penalty: `-0.1` per conflict
- Missing penalty: `-0.05` per missing field

### Temporal Scaling
The temporal score reduces confidence when the two modalities are less well aligned in time.

The final result is clamped to: `[0.0, 1.0]`.

## Fusion Output

The fusion process produces a `FusionResult` containing:

- Final `RobotCommand`
- Field-level provenance (voice / gesture / agreement)
- Conflict fields
- Temporal metadata
- Confidence decision reasoning
- Ambiguity flags

This supports both:
- Real-time feedback
- Detailed post-hoc analysis


## Worked Example 1:  Complementary Fusion

### Input

#### Voice:
```
"pick up the red cube"
-> action = pick (0.85)
-> object = red_cube (0.80)
-> location = None
```

#### Gesture:
```
hand positioned left
-> location = left (0.75)
```

### Fusion
- Temporal alignment is within the fusion window
- No conflicts between modalities
- The missing `location` field is filled using gesture input
- All fields meet their respective confidence thresholds

### Output
```
action = pick
object = red_cube
location = left
confidence = high
```

The final confidence in this case is high due to:
- strong voice confidence for action and object
- successful completion of missing fields via gesture
- absence of conflicts or penalties
- temporal alignment within the fusion window

## Worked Example 2: Conflict and Ambiguity

### Input

#### Voice:
```
"place the red cube on the left"
-> action = place (0.78)
-> object = red_cube (0.82)
-> location = left (0.60)
```

#### Gesture:
```
hand positioned right
-> location = right (0.65)
```

### Fusion

- Temporal alignment is within the fusion window
- Conflict detected in `location` field:
  - voice = left (0.60)
  - gesture = right (0.65)

- Confidence comparison:
  - gesture confidence exceeds voice confidence beyond the bias margin

- Therefore, gesture overrides voice for the `location` field

- Confidence gap is small:

$$
|\text{voice}_{conf} - \text{gesture}_{conf}| < 0.15
$$

- Result is flagged as **ambiguous**

### Output
```
action = place
object = red_cube
location = right
confidence = medium
```

The final confidence is reduced due to:
- conflict penalty
- small confidence gap (ambiguity)

---