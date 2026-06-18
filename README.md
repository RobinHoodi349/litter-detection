# First Lab KI-Systeme

Over all task is to build a robot that can detect litter and notify its operator.

This project was build with the autoresaerch idea of Andrew Karpathy: https://github.com/karpathy/autoresearch

The overall idea is to critically look at the experiments and progress the AI made, identify improvements and integrate a further improved version into a robot setup.

Other approaches fine-tune a yolo model: e.g. see for https://github.com/jeremy-rico/litter-detection

## 1 Student Task

- [Task Description](docs/student_task.md)
- [Context to this project](docs/explainer.md)

## 2 Student Task

- [Task Description 2](docs/student_task_2.md)

## Example images not in the dataset

|No litter | Litter |
|---|---|
|![](docs/images/Image2.jpeg) | ![](docs/images/Image3.jpeg) |

## Autoresearch Content

> Note: There is already one good model in this repository. Thus you should be able to investigate the performance using the Analysis Notebook.

- [Analysis Notebook](auto-research/analysis.ipynb)
- [Instructions](auto-research/program.md)
- [Finding from previous runs](auto-research/findings.md)

## Setup

Init project (desktop: Windows AMD64 / Linux x86_64):

```bash
uv sync
```

`uv sync`/`uv run` work without the `vendor/` directory on desktop platforms —
the lockfile is resolved only for the desktop environments (see `[tool.uv]
environments` in `pyproject.toml`), so the aarch64 Jetson wheels are never
referenced off the Jetson.

### Jetson setup (aarch64)

The Jetson is intentionally **not** covered by the desktop lockfile, because its
CUDA-enabled `torch`/`torchvision` come from NVIDIA JetPack builds that aren't on
PyPI. Install the CUDA wheels first (from the vendored files, transferred to the
Jetson separately — they're too large for GitHub and are git-ignored), then the
project on top so the `torch>=2.3.0` requirement is already satisfied:

```bash
uv venv --python 3.11
uv pip install vendor/jetson-wheels/*.whl   # CUDA torch + torchvision (JetPack)
uv pip install -e . --no-deps               # the project itself
uv pip install eclipse-zenoh opencv-python-headless pydantic pydantic-ai \
  opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc python-dotenv
# vendor/pyrealsense2 is already importable from its build/ dir on the Jetson
```

(The last line installs only the runtime deps the robot actually needs; adjust to
taste. The point is that `torch`/`torchvision` come from the vendored wheels, not
from a `path` source in `pyproject.toml`.) `vendor/pyrealsense2` is a Jetson-only
RealSense build and is not used off the Jetson — it's referenced by neither uv nor
any Python import.

Content:

- There is an [analysis notebook](auto-research/analysis.ipynb) to take a first look on the project and test the existing models.
- The project contains a mlflow project that stores the hole experiment and training history.
  Run the following command to launch the mlflow server and ui
  ```bash
  uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --default-artifact-root ./artifacts/mlflow/mlruns
  ```





## Additional Content

- [Experiment Tracking](https://mlflow.org/docs/latest/ml/getting-started/deep-learning/)
