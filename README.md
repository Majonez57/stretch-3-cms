# Multimodal HRI Command Hub

A multimodal human–robot interaction system that combines voice and gesture input to enable intuitive command execution for assistive robotic tasks.

The system supports three interaction modes:
- Voice input (natural language)
- Gesture input (hand gestures)
- Multimodal input (fused voice and gesture)

This project was developed as part of a final year Computer Science dissertation focusing on usability, performance, and multimodal fusion.


![Multimodal HRI Command Hub UI](/assets/live-interaction-screenshot.png)
*Figure: Live interaction interface showing multimodal command input and system output.*

## System Overview

The system consists of:
- Voice module: speech-to-text and command parsing
- Gesture module: hand tracking and gesture recognition
- Fusion module: combines inputs using temporal and confidence-based logic
- UI: Streamlit-based interface for interaction and experiments
- Analysis pipeline: evaluates performance using logged data

For detailed design see:
- [System Architecture](./docs/system_architecture.md)
- [Fusion Design](./docs/fusion_design.md)
- [Experiment Design](./docs/experiment_design.md)
- [Analysis](./docs/analysis.md)

## Installation

Clone the repository:

```bash
git clone https://github.com/rubenodamo/multimodal-hri.git
cd multimodal-hri

# Install dependencies
pip install -r requirements.txt
```

## Running the Application
Start the Streamlit app:

```bash
streamlit run app.py
```

This launches the interface with:
- Live interaction mode
- Experiment mode for data collection

## Running Tests

Run the test suite using pytest:

```bash
pytest
```

Optional:

```bash
# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_fusion.py
```


## Running the Analysis

The analysis pipeline can be executed via command line:

```bash
python -m analysis.run_analysis

# Analyse a specific file
python -m analysis.run_analysis --file path/to/session.csv

# Analyse all logs in a directory
python -m analysis.run_analysis --dir logs/

# Save plots to disk
python -m analysis.run_analysis --save-plots

# Custom output directory
python -m analysis.run_analysis --output-dir my/plots/
```

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
│   └── run_analysis.py
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
│   └── system_architecture.md
│ 
├── assets                          # Images
│
└── logs                            # Logs (generated at runtime)
```

## Technologies

- Python
- Streamlit
- MediaPipe (gesture recognition)
- OpenAI Whisper (speech recognition)
- pandas (data analysis)
- matplotlib (visualisation)

## Code Quality & Tooling

| Tool                                    | Purpose       | Config           |
|-----------------------------------------|---------------|------------------|
| [Black](https://black.readthedocs.io)   | Formatter     | `pyproject.toml` |
| [isort](https://pycqa.github.io/isort/) | Import sorter | `pyproject.toml` |
| [pylint](https://pylint.readthedocs.io) | Linter        | `.pylintrc`      |

### Running the tools

```bash
# Format
black .
isort .

# Lint
pylint app.py config.py models.py voice/ gesture/ fusion/ trial_logger/ experiments/ analysis/ ui/

# Pre-commit (one-time setup)
pip install pre-commit
pre-commit install
```

## Notes

- The `logs/` directory is created automatically at runtime and is not included in the repository.
- Webcam access is required for gesture input.
- Microphone access is required for voice input.
