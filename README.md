# Cambria Interactive

**Cambria Interactive**, is bulit for interactive **[Cambria](https://github.com/cinescope-wkr/cambria)** ecosystem. 

[Artificial Cambrian Intelligence (ACI)](https://github.com/cambrian-org/ACI) has inspired this project substantially, which is a simulation framework for **co-designing visual morphology and control policies** in embodied agents.
It combines MuJoCo physics, configurable eye models, reinforcement learning, and evolutionary search to study how task demands and environment visibility shape vision.

This repository includes both:
- Research-grade training/evaluation pipelines.
- Production-style experiment scripts for multi-run tracking and cinematic rendering.

## Research Scope

ACI is designed to answer questions such as:
- Which eye morphology is advantageous for a given task (navigation, detection, tracking)?
- When does broader visual coverage outperform optics realism (or vice versa)?
- How robust are learned policies across environment visibility regimes (clear, shallow haze, deep sea)?

## Core Capabilities

- **Tasks**: `detection`, `tracking`, `navigation`.
- **Eye models**:
  - `eye` (single pinhole-like eye)
  - `multi_eye` (compound/distributed eyes)
  - `optics` (wave/geometric blur with aperture + PSF)
  - `cinematic_optics` (constrained optical genome + depth-dependent cinematic optics)
- **Renderer presets**:
  - `renderer` (default/clear)
  - `cambrian_shallows`
  - `deep_sea`
  - `cinematic`
  - `cambrian_shallows_showcase`
  - `cambrian_shallows_showcase_scene`
  - plus utilities: `tracking`, `bev`, `bev_tracking`, `fixed`
- **Recording sources**:
  - `scene` (default)
  - `agent_vision`
  - `side_by_side`
- **Output save modes**: `WEBP`, `MP4`, `GIF`, `PNG`, `USD`.
- **Optimization modes**:
  - Standard RL training/evaluation (Stable-Baselines3 PPO stack)
  - Evolutionary sweeps via Hydra + Nevergrad
- **Batch orchestration**:
  - tmux-based multi-GPU scripts for morphology suites
  - checkpoint-lineage rendering and cinematic showcase rendering

## Project Papers

- *What if Eye...? Computationally Recreating Vision Evolution*  
  [Paper](https://arxiv.org/pdf/2501.15001) | [Website](https://eyes.mit.edu) | [Documentation](https://eyes.mit.edu/ACI/)
- *A Roadmap for Generative Design of Visual Intelligence*  
  [Paper](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3) | [Documentation](https://eyes.mit.edu/ACI/)

## Installation

### Prerequisites

- Python `>=3.11,<4.0`
- MuJoCo-compatible OpenGL runtime (`egl` for headless training is recommended)
- Java (required by current Hydra setup in this codebase)

### Setup

```bash
git clone https://github.com/cambrian-org/ACI.git
cd ACI
pip install -e .
```

For docs/dev extras:

```bash
pip install -e '.[doc,dev]'
```

## Execution Modes

### 1) Train

```bash
bash scripts/run.sh cambrian/main.py --train example=detection
```

### 2) Evaluate (trained checkpoint)

```bash
bash scripts/run.sh cambrian/main.py --eval example=detection trainer/model=loaded_model
```

### 3) Evaluate privileged control baseline

```bash
bash scripts/run.sh cambrian/main.py --eval example=detection env/agents@env.agents.agent=point_seeker
```

### 4) Evolutionary optimization

```bash
bash scripts/run.sh cambrian/main.py --train task=detection evo=evo -m
```

Use `evo/mutations` and `evo/constraints` config groups to control mutation space and feasibility constraints.

## Task and Environment Configuration

### Available tasks

- `task=detection`: discriminate goal vs adversary textures.
- `task=tracking`: moving goal/adversary (detection dynamics + moving targets).
- `task=navigation`: maze traversal with wall-contact penalties.

### Environment/renderer presets

Set with `env/renderer=<preset>`.

- `renderer`: default control condition (clear/basic).
- `cambrian_shallows`: shallow-water haze/fog + lighting.
- `deep_sea`: lower-visibility deep-sea lighting/haze profile.
- `cinematic`: opt-in cinematic post-FX and camera preset.
- `cambrian_shallows_showcase`: higher-fidelity showcase scene and style.
- `cambrian_shallows_showcase_scene`: fixed wide scene camera for showcase outputs.

### Recording modes

Set with `record_source=<mode>` (during eval):

- `scene`
- `agent_vision`
- `side_by_side`

Example (tracking eval, side-by-side mp4):

```bash
bash scripts/run.sh cambrian/main.py --eval \
  task=tracking \
  trainer/model=loaded_model \
  env/renderer=cambrian_shallows \
  record_source=side_by_side \
  +eval_env.renderer.save_mode=MP4
```

## Eye/Morphology Configuration

Common overrides:

```bash
# Base single eye
env/agents/eyes@env.agents.agent.eyes.eye=eye

# Multi-eye
env/agents/eyes@env.agents.agent.eyes.eye=multi_eye

# Optics eye
env/agents/eyes@env.agents.agent.eyes.eye=optics

# Cinematic optics eye
env/agents/eyes@env.agents.agent.eyes.eye=cinematic_optics
```

Typical fields:
- `env.agents.agent.eyes.eye.resolution=[H,W]`
- `env.agents.agent.eyes.eye.fov=[hdeg,vdeg]`
- `env.agents.agent.eyes.eye.num_eyes=[min,max]` (multi-eye)
- `env.agents.agent.eyes.eye.lon_range=[min,max]`, `lat_range=[min,max]`
- `env.agents.agent.eyes.eye.aperture.radius=<float>` (optics)

## Built-in Experiment Pipelines

### A) Tracking morphology suite (2-GPU tmux pipeline)

Script: `scripts/run_vast2_tracking_eyes.sh`

Runs four variants:
- Base eye
- Multi-eye
- Optics eye
- Narrow lens (optics with narrow FOV)

Default behavior:
- Two GPU pipelines (`0`: base→multi, `1`: optics→narrow)
- Periodic evaluation checkpoints
- Automatic `agent_vision` eval after each completed training

Commands:

```bash
./scripts/run_vast2_tracking_eyes.sh up
./scripts/run_vast2_tracking_eyes.sh status
./scripts/run_vast2_tracking_eyes.sh tail
./scripts/run_vast2_tracking_eyes.sh eval-all
./scripts/run_vast2_tracking_eyes.sh scene-eval-all
./scripts/run_vast2_tracking_eyes.sh render-all
./scripts/run_vast2_tracking_eyes.sh stop
```

### B) Tracking shallow-water full suite (Experiment A)

Script: `scripts/run_vast2_tracking_eyes_shallows.sh`

This is a wrapper of the above suite with:
- `EXPERIMENT_RENDERER=cambrian_shallows`
- `AUTO_SCENE_EVAL=1`
- scene eval output defaulted to MP4

Commands are the same as the base suite:

```bash
./scripts/run_vast2_tracking_eyes_shallows.sh up
./scripts/run_vast2_tracking_eyes_shallows.sh status
./scripts/run_vast2_tracking_eyes_shallows.sh tail
./scripts/run_vast2_tracking_eyes_shallows.sh scene-eval-all
./scripts/run_vast2_tracking_eyes_shallows.sh render-all
```

### C) Cinematic dual-task run

Script: `scripts/run_vast2_cinematic.sh`

Default pairing:
- GPU 0: detection with `env/renderer=deep_sea`
- GPU 1: navigation with `env/renderer=cambrian_shallows`

Useful commands:

```bash
./scripts/run_vast2_cinematic.sh up
./scripts/run_vast2_cinematic.sh status
./scripts/run_vast2_cinematic.sh tail
./scripts/run_vast2_cinematic.sh eval-all
./scripts/run_vast2_cinematic.sh stop
```

## Render-Only Cambrian Showcase

Script: `scripts/render_cambrian_showcase.sh`

Purpose:
- Keep learned policy fixed.
- Upgrade visual world (cinematic shallow sea, themed agents/materials, scene styling).
- Produce publication/demo media without retraining.

Usage:

```bash
./scripts/render_cambrian_showcase.sh <variant> <logdir>
```

Variants:
- `base`
- `multi`
- `optics`
- `narrow`

Modes:
- `SHOWCASE_MODE=faithful`: faithful replay in shallow renderer.
- `SHOWCASE_MODE=showcase` (default): cinematic showcase renderer stack.

Generated artifacts per run include:
- `cambrian_<mode>_scene.*`
- `cambrian_<mode>_agent_vision.*`
- `cambrian_<mode>_side_by_side.*`

## Outputs and Artifacts

Each experiment directory (typically under `logs/<date>/<expname>` or custom `logdir`) can include:

- `best_model.zip`: best checkpoint by eval callback
- `policy.pt`: exported policy weights
- `monitor.csv`: training monitor
- `eval_monitor.csv`: evaluation monitor
- `train_fitness.txt`
- `<eval_name>_fitness.txt` (e.g., `eval_agent_vision_fitness.txt`)
- `evaluations.npz`
- `evaluations/monitor.png`, `evaluations/eval_monitor.png`
- renderer outputs (`.webp`, `.mp4`, `.gif`, `.png`, optional `.usd`)
- optional checkpoint lineage under `checkpoints/`

## Reproducibility and Practical Notes

- Use `MUJOCO_GL=egl` for headless training speed.
- `run.sh` is a convenient wrapper; direct `python cambrian/main.py ...` also works.
- Training may stop early by design due to no-improvement stopping callback (`max_no_improvement_evals=3`).
- In this branch, WandB toggles exist in scripts but no active WandB integration is wired in config/loggers.
- For learning-curve interpretation, prefer the plotted evaluation artifacts (`evaluations/*.png`) in addition to final post-hoc eval outputs.

## Documentation

- Full docs: https://eyes.mit.edu/ACI/
- Local docs build:

```bash
cd docs
make clean html
```

## Citation

```bibtex
@software{aci,
    author = {Aaron Young and Kushagra Tiwary and Zaid Tasneem and Tzofi Klinghoffer and Bhavya Agrawalla and Sanjana Duttagupta and Akshat Dave and Brian Cheung},
    title = {{Artificial Cambrian Intelligence}},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/cambrian-org/ACI}},
}
```
