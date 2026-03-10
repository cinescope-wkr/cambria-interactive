#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-tracking_eyes}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run.sh}"
PYTHON_BIN="${PYTHON_BIN:-}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/vast2_tracking}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/logs/vast2_tracking_runs}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${RUNS_ROOT}/${RUN_TIMESTAMP}}"
STATE_DIR="${STATE_DIR:-${RUNS_ROOT}/.state}"
CURRENT_RUN_FILE="${CURRENT_RUN_FILE:-${STATE_DIR}/${SESSION_NAME}.env}"

MUJOCO_GL_VALUE="${MUJOCO_GL:-egl}"
MPLCONFIGDIR_VALUE="${MPLCONFIGDIR:-/tmp/mpl}"
JOBLIB_TEMP_FOLDER_VALUE="${JOBLIB_TEMP_FOLDER:-/tmp}"
XDG_CACHE_HOME_VALUE="${XDG_CACHE_HOME:-/tmp/.cache}"
MESA_SHADER_CACHE_DIR_VALUE="${MESA_SHADER_CACHE_DIR:-/tmp/mesa_shader_cache}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
TRACKING_EVAL_EPISODES="${TRACKING_EVAL_EPISODES:-1}"
EVAL_SAVE_MODE="${EVAL_SAVE_MODE:-WEBP}"
EXPERIMENT_RENDERER="${EXPERIMENT_RENDERER:-renderer}"
AUTO_SCENE_EVAL="${AUTO_SCENE_EVAL:-0}"
SCENE_EVAL_SAVE_MODE="${SCENE_EVAL_SAVE_MODE:-MP4}"
USE_TENSORBOARD="${USE_TENSORBOARD:-0}"
USE_WANDB="${USE_WANDB:-0}"
CHECKPOINT_DIRNAME="${CHECKPOINT_DIRNAME:-checkpoints}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-checkpoint}"
CHECKPOINT_MAX_KEEP="${CHECKPOINT_MAX_KEEP:-0}"
RENDER_FPS="${RENDER_FPS:-20}"
RENDER_LAYOUT="${RENDER_LAYOUT:-side_by_side}"
RENDER_SAVE_MODE="${RENDER_SAVE_MODE:-MP4}"

BASE_EXPNAME="${BASE_EXPNAME:-vast2_tracking_base_eye}"
MULTI_EXPNAME="${MULTI_EXPNAME:-vast2_tracking_multi_eye}"
OPTICS_EXPNAME="${OPTICS_EXPNAME:-vast2_tracking_optics_eye}"
NARROW_EXPNAME="${NARROW_EXPNAME:-vast2_tracking_narrow_lens}"

BASE_LOGDIR="${BASE_LOGDIR:-${LOG_ROOT}/${BASE_EXPNAME}}"
MULTI_LOGDIR="${MULTI_LOGDIR:-${LOG_ROOT}/${MULTI_EXPNAME}}"
OPTICS_LOGDIR="${OPTICS_LOGDIR:-${LOG_ROOT}/${OPTICS_EXPNAME}}"
NARROW_LOGDIR="${NARROW_LOGDIR:-${LOG_ROOT}/${NARROW_EXPNAME}}"

GPU0_LOG_FILE="${GPU0_LOG_FILE:-${RUN_LOG_DIR}/gpu0_base_multi.log}"
GPU1_LOG_FILE="${GPU1_LOG_FILE:-${RUN_LOG_DIR}/gpu1_optics_narrow.log}"

BASE_TENSORBOARD_DIR="${BASE_TENSORBOARD_DIR:-${RUN_LOG_DIR}/tensorboard/base}"
MULTI_TENSORBOARD_DIR="${MULTI_TENSORBOARD_DIR:-${RUN_LOG_DIR}/tensorboard/multi}"
OPTICS_TENSORBOARD_DIR="${OPTICS_TENSORBOARD_DIR:-${RUN_LOG_DIR}/tensorboard/optics}"
NARROW_TENSORBOARD_DIR="${NARROW_TENSORBOARD_DIR:-${RUN_LOG_DIR}/tensorboard/narrow}"

resolve_python_bin() {
  if [ -n "${PYTHON_BIN}" ] && [ -x "${PYTHON_BIN}" ]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi

  if [ -x /venv/main/bin/python ]; then
    printf '%s\n' /venv/main/bin/python
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "Could not find a usable python interpreter. Set PYTHON_BIN explicitly." >&2
  exit 1
}

require_tmux() {
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is required but not installed." >&2
    exit 1
  }
}

require_runner_script() {
  [ -f "${RUNNER_SCRIPT}" ] || {
    echo "Could not find runner script at ${RUNNER_SCRIPT}" >&2
    exit 1
  }
}

warn_wandb_if_requested() {
  if [ "${USE_WANDB}" = "1" ]; then
    echo "USE_WANDB=1 was requested, but this repo does not expose a WandB Hydra/logger integration. Ignoring." >&2
  fi
}

session_exists() {
  tmux has-session -t "${SESSION_NAME}" 2>/dev/null
}

ensure_dirs() {
  mkdir -p \
    "${LOG_ROOT}" \
    "${RUNS_ROOT}" \
    "${RUN_LOG_DIR}" \
    "${STATE_DIR}" \
    "${BASE_LOGDIR}" \
    "${MULTI_LOGDIR}" \
    "${OPTICS_LOGDIR}" \
    "${NARROW_LOGDIR}" \
    "${BASE_TENSORBOARD_DIR}" \
    "${MULTI_TENSORBOARD_DIR}" \
    "${OPTICS_TENSORBOARD_DIR}" \
    "${NARROW_TENSORBOARD_DIR}" \
    "${MPLCONFIGDIR_VALUE}" \
    "${XDG_CACHE_HOME_VALUE}" \
    "${MESA_SHADER_CACHE_DIR_VALUE}" \
    "$(dirname "${GPU0_LOG_FILE}")" \
    "$(dirname "${GPU1_LOG_FILE}")"
}

common_exports() {
  cat <<EOF
export MUJOCO_GL=$(printf '%q' "${MUJOCO_GL_VALUE}")
export MPLCONFIGDIR=$(printf '%q' "${MPLCONFIGDIR_VALUE}")
export JOBLIB_TEMP_FOLDER=$(printf '%q' "${JOBLIB_TEMP_FOLDER_VALUE}")
export XDG_CACHE_HOME=$(printf '%q' "${XDG_CACHE_HOME_VALUE}")
export MESA_SHADER_CACHE_DIR=$(printf '%q' "${MESA_SHADER_CACHE_DIR_VALUE}")
export PATH=$(printf '%q' "$(dirname "${PYTHON_BIN}")"):\$PATH
cd $(printf '%q' "${REPO_ROOT}")
mkdir -p "\$MPLCONFIGDIR" "\$XDG_CACHE_HOME" "\$MESA_SHADER_CACHE_DIR" \
  $(printf '%q' "${RUN_LOG_DIR}") \
  $(printf '%q' "${BASE_LOGDIR}") \
  $(printf '%q' "${MULTI_LOGDIR}") \
  $(printf '%q' "${OPTICS_LOGDIR}") \
  $(printf '%q' "${NARROW_LOGDIR}") \
  $(printf '%q' "${BASE_TENSORBOARD_DIR}") \
  $(printf '%q' "${MULTI_TENSORBOARD_DIR}") \
  $(printf '%q' "${OPTICS_TENSORBOARD_DIR}") \
  $(printf '%q' "${NARROW_TENSORBOARD_DIR}")
EOF
}

write_current_run_state() {
  mkdir -p "${STATE_DIR}"
  cat > "${CURRENT_RUN_FILE}" <<EOF
RUN_TIMESTAMP=$(printf '%q' "${RUN_TIMESTAMP}")
RUN_LOG_DIR=$(printf '%q' "${RUN_LOG_DIR}")
GPU0_LOG_FILE=$(printf '%q' "${GPU0_LOG_FILE}")
GPU1_LOG_FILE=$(printf '%q' "${GPU1_LOG_FILE}")
TOTAL_TIMESTEPS=$(printf '%q' "${TOTAL_TIMESTEPS}")
EVAL_FREQ=$(printf '%q' "${EVAL_FREQ}")
TRACKING_EVAL_EPISODES=$(printf '%q' "${TRACKING_EVAL_EPISODES}")
EVAL_SAVE_MODE=$(printf '%q' "${EVAL_SAVE_MODE}")
EXPERIMENT_RENDERER=$(printf '%q' "${EXPERIMENT_RENDERER}")
AUTO_SCENE_EVAL=$(printf '%q' "${AUTO_SCENE_EVAL}")
SCENE_EVAL_SAVE_MODE=$(printf '%q' "${SCENE_EVAL_SAVE_MODE}")
USE_TENSORBOARD=$(printf '%q' "${USE_TENSORBOARD}")
USE_WANDB=$(printf '%q' "${USE_WANDB}")
CHECKPOINT_DIRNAME=$(printf '%q' "${CHECKPOINT_DIRNAME}")
CHECKPOINT_PREFIX=$(printf '%q' "${CHECKPOINT_PREFIX}")
CHECKPOINT_MAX_KEEP=$(printf '%q' "${CHECKPOINT_MAX_KEEP}")
RENDER_FPS=$(printf '%q' "${RENDER_FPS}")
RENDER_LAYOUT=$(printf '%q' "${RENDER_LAYOUT}")
RENDER_SAVE_MODE=$(printf '%q' "${RENDER_SAVE_MODE}")
RUNNER_SCRIPT=$(printf '%q' "${RUNNER_SCRIPT}")
BASE_EXPNAME=$(printf '%q' "${BASE_EXPNAME}")
MULTI_EXPNAME=$(printf '%q' "${MULTI_EXPNAME}")
OPTICS_EXPNAME=$(printf '%q' "${OPTICS_EXPNAME}")
NARROW_EXPNAME=$(printf '%q' "${NARROW_EXPNAME}")
BASE_LOGDIR=$(printf '%q' "${BASE_LOGDIR}")
MULTI_LOGDIR=$(printf '%q' "${MULTI_LOGDIR}")
OPTICS_LOGDIR=$(printf '%q' "${OPTICS_LOGDIR}")
NARROW_LOGDIR=$(printf '%q' "${NARROW_LOGDIR}")
BASE_TENSORBOARD_DIR=$(printf '%q' "${BASE_TENSORBOARD_DIR}")
MULTI_TENSORBOARD_DIR=$(printf '%q' "${MULTI_TENSORBOARD_DIR}")
OPTICS_TENSORBOARD_DIR=$(printf '%q' "${OPTICS_TENSORBOARD_DIR}")
NARROW_TENSORBOARD_DIR=$(printf '%q' "${NARROW_TENSORBOARD_DIR}")
PYTHON_BIN=$(printf '%q' "${PYTHON_BIN}")
EOF
}

load_current_run_state() {
  if [ ! -f "${CURRENT_RUN_FILE}" ]; then
    echo "No current run metadata found at ${CURRENT_RUN_FILE}. Start a run first." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  . "${CURRENT_RUN_FILE}"
}

build_control_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
cat <<'MSG'
Session: ${SESSION_NAME}
Run timestamp: ${RUN_TIMESTAMP}

Windows:
  - gpu0_pipeline: GPU 0 runs Base eye -> Multi-eye
  - gpu1_pipeline: GPU 1 runs Optics eye -> Narrow lens

Useful commands from another shell:
  ./scripts/run_vast2_tracking_eyes.sh attach
  ./scripts/run_vast2_tracking_eyes.sh status
  ./scripts/run_vast2_tracking_eyes.sh tail
  ./scripts/run_vast2_tracking_eyes.sh eval-all
  ./scripts/run_vast2_tracking_eyes.sh render-all
  ./scripts/run_vast2_tracking_eyes.sh stop

Experiment outputs:
  Base   : ${BASE_LOGDIR}
  Multi  : ${MULTI_LOGDIR}
  Optics : ${OPTICS_LOGDIR}
  Narrow : ${NARROW_LOGDIR}

Pipeline logs:
  GPU 0: ${GPU0_LOG_FILE}
  GPU 1: ${GPU1_LOG_FILE}

Timesteps per experiment: ${TOTAL_TIMESTEPS}
Eval frequency: ${EVAL_FREQ}
Renderer: ${EXPERIMENT_RENDERER}
Eval save mode: ${EVAL_SAVE_MODE}
Automatic scene eval: ${AUTO_SCENE_EVAL} (save mode: ${SCENE_EVAL_SAVE_MODE})
Checkpoint snapshots: every eval -> ${CHECKPOINT_DIRNAME}/${CHECKPOINT_PREFIX}_*.zip
Checkpoint max_keep: ${CHECKPOINT_MAX_KEEP}
Render layout default: ${RENDER_LAYOUT}
Render save mode default: ${RENDER_SAVE_MODE}
TensorBoard: ${USE_TENSORBOARD}
WandB: ${USE_WANDB} (not wired in this repo)
MSG
exec bash
EOF
}

build_pipeline_common() {
  cat <<EOF
append_checkpoint_callback_overrides() {
  local logdir="\$1"
  local arr_name="\$2"

  eval "\${arr_name}+=(\"+trainer.callbacks.eval_callback.callback_after_eval.callbacks.periodic_checkpoint_callback._target_=cambrian.ml.callbacks.MjCambrianPeriodicCheckpointCallback\")"
  eval "\${arr_name}+=(\"+trainer.callbacks.eval_callback.callback_after_eval.callbacks.periodic_checkpoint_callback.logdir=\${logdir}\")"
  eval "\${arr_name}+=(\"+trainer.callbacks.eval_callback.callback_after_eval.callbacks.periodic_checkpoint_callback.checkpoint_dirname=${CHECKPOINT_DIRNAME}\")"
  eval "\${arr_name}+=(\"+trainer.callbacks.eval_callback.callback_after_eval.callbacks.periodic_checkpoint_callback.prefix=${CHECKPOINT_PREFIX}\")"
  eval "\${arr_name}+=(\"+trainer.callbacks.eval_callback.callback_after_eval.callbacks.periodic_checkpoint_callback.max_keep=${CHECKPOINT_MAX_KEEP}\")"
}

run_training_then_eval() {
  local label="\$1"
  local gpu_id="\$2"
  local expname="\$3"
  local logdir="\$4"
  local tbdir="\$5"
  shift 5
  local overrides=("\$@")

  mkdir -p "\$logdir" "\$tbdir"

  echo ""
  echo "===== [\${label}] training start ====="
  echo "[\${label}] expname=\${expname}"
  echo "[\${label}] logdir=\${logdir}"
  echo "[\${label}] total_timesteps=${TOTAL_TIMESTEPS} eval_freq=${EVAL_FREQ}"

  local train_cmd=(
    env
    CUDA_VISIBLE_DEVICES="\${gpu_id}"
    bash
    $(printf '%q' "${RUNNER_SCRIPT}")
    cambrian/main.py
    --train
    task=tracking
    trainer.total_timesteps=${TOTAL_TIMESTEPS}
    env/renderer=${EXPERIMENT_RENDERER}
    trainer.callbacks.eval_callback.eval_freq=${EVAL_FREQ}
    trainer.callbacks.eval_callback.render=false
    env.add_overlays=false
    env.debug_overlays_size=0
    expname="\${expname}"
    "logdir=\${logdir}"
  )
  train_cmd+=("\${overrides[@]}")
  append_checkpoint_callback_overrides "\${logdir}" train_cmd

  if [ "${USE_TENSORBOARD}" = "1" ]; then
    train_cmd+=("+trainer.model.tensorboard_log=\${tbdir}")
  fi

  echo "[\${label}] command: \${train_cmd[*]}"
  "\${train_cmd[@]}"

  if [ -f "\${logdir}/best_model.zip" ]; then
    echo "[\${label}] training complete, running agent_vision eval"
    local eval_cmd=(
      env
      CUDA_VISIBLE_DEVICES="\${gpu_id}"
      bash
      $(printf '%q' "${RUNNER_SCRIPT}")
      cambrian/main.py
      --eval
      task=tracking
      env/renderer=${EXPERIMENT_RENDERER}
      record_source=agent_vision
      trainer/model=loaded_model
      env.add_overlays=false
      env.debug_overlays_size=0
      expname="\${expname}"
      "logdir=\${logdir}"
      eval_env.n_eval_episodes=${TRACKING_EVAL_EPISODES}
      eval_env.save_filename=eval_agent_vision
      +eval_env.renderer.save_mode=${EVAL_SAVE_MODE}
    )
    eval_cmd+=("\${overrides[@]}")
    echo "[\${label}] command: \${eval_cmd[*]}"
    "\${eval_cmd[@]}"

    if [ "${AUTO_SCENE_EVAL}" = "1" ]; then
      echo "[\${label}] running scene eval"
      local scene_eval_cmd=(
        env
        CUDA_VISIBLE_DEVICES="\${gpu_id}"
        bash
        $(printf '%q' "${RUNNER_SCRIPT}")
        cambrian/main.py
        --eval
        task=tracking
        env/renderer=${EXPERIMENT_RENDERER}
        trainer/model=loaded_model
        env.add_overlays=false
        env.debug_overlays_size=0
        expname="\${expname}"
        "logdir=\${logdir}"
        eval_env.n_eval_episodes=${TRACKING_EVAL_EPISODES}
        eval_env.save_filename=eval_scene
        +eval_env.renderer.save_mode=${SCENE_EVAL_SAVE_MODE}
      )
      scene_eval_cmd+=("\${overrides[@]}")
      echo "[\${label}] command: \${scene_eval_cmd[*]}"
      "\${scene_eval_cmd[@]}"
    fi
  else
    echo "[\${label}] best_model.zip not found, skipping automatic eval" >&2
  fi

  echo "===== [\${label}] done ====="
}

run_eval_only() {
  local label="\$1"
  local gpu_id="\$2"
  local expname="\$3"
  local logdir="\$4"
  shift 4
  local overrides=("\$@")

  if [ ! -f "\${logdir}/best_model.zip" ]; then
    echo "[\${label}] best_model.zip not found at \${logdir}" >&2
    return 0
  fi

  echo ""
  echo "===== [\${label}] manual eval start ====="
  local eval_cmd=(
    env
    CUDA_VISIBLE_DEVICES="\${gpu_id}"
    bash
    $(printf '%q' "${RUNNER_SCRIPT}")
    cambrian/main.py
    --eval
    task=tracking
    env/renderer=${EXPERIMENT_RENDERER}
    record_source=agent_vision
    trainer/model=loaded_model
    env.add_overlays=false
    env.debug_overlays_size=0
    expname="\${expname}"
    "logdir=\${logdir}"
    eval_env.n_eval_episodes=${TRACKING_EVAL_EPISODES}
    eval_env.save_filename=eval_agent_vision_latest
    +eval_env.renderer.save_mode=${EVAL_SAVE_MODE}
  )
  eval_cmd+=("\${overrides[@]}")
  echo "[\${label}] command: \${eval_cmd[*]}"
  "\${eval_cmd[@]}"
  echo "===== [\${label}] manual eval done ====="
}

run_scene_eval_only() {
  local label="\$1"
  local gpu_id="\$2"
  local expname="\$3"
  local logdir="\$4"
  shift 4
  local overrides=("\$@")

  if [ ! -f "\${logdir}/best_model.zip" ]; then
    echo "[\${label}] best_model.zip not found at \${logdir}" >&2
    return 0
  fi

  echo ""
  echo "===== [\${label}] manual scene eval start ====="
  local eval_cmd=(
    env
    CUDA_VISIBLE_DEVICES="\${gpu_id}"
    bash
    $(printf '%q' "${RUNNER_SCRIPT}")
    cambrian/main.py
    --eval
    task=tracking
    env/renderer=${EXPERIMENT_RENDERER}
    trainer/model=loaded_model
    env.add_overlays=false
    env.debug_overlays_size=0
    expname="\${expname}"
    "logdir=\${logdir}"
    eval_env.n_eval_episodes=${TRACKING_EVAL_EPISODES}
    eval_env.save_filename=eval_scene_latest
    +eval_env.renderer.save_mode=${SCENE_EVAL_SAVE_MODE}
  )
  eval_cmd+=("\${overrides[@]}")
  echo "[\${label}] command: \${eval_cmd[*]}"
  "\${eval_cmd[@]}"
  echo "===== [\${label}] manual scene eval done ====="
}
EOF
}

build_gpu0_pipeline_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU0_LOG_FILE}")) 2>&1
$(build_pipeline_common)

echo "[gpu0_pipeline] starting Base eye -> Multi-eye sequence"
echo "[gpu0_pipeline] run_timestamp=${RUN_TIMESTAMP}"
echo "[gpu0_pipeline] log_file=${GPU0_LOG_FILE}"

run_training_then_eval \
  "base_eye" \
  "0" \
  "${BASE_EXPNAME}" \
  $(printf '%q' "${BASE_LOGDIR}") \
  $(printf '%q' "${BASE_TENSORBOARD_DIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

run_training_then_eval \
  "multi_eye" \
  "0" \
  "${MULTI_EXPNAME}" \
  $(printf '%q' "${MULTI_LOGDIR}") \
  $(printf '%q' "${MULTI_TENSORBOARD_DIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.num_eyes=[1,3]" \
  "env.agents.agent.eyes.eye.lon_range=[-30,30]" \
  "env.agents.agent.eyes.eye.lat_range=[-5,5]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

echo "[gpu0_pipeline] finished"
exec bash
EOF
}

build_gpu1_pipeline_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU1_LOG_FILE}")) 2>&1
$(build_pipeline_common)

echo "[gpu1_pipeline] starting Optics eye -> Narrow lens sequence"
echo "[gpu1_pipeline] run_timestamp=${RUN_TIMESTAMP}"
echo "[gpu1_pipeline] log_file=${GPU1_LOG_FILE}"

run_training_then_eval \
  "optics_eye" \
  "1" \
  "${OPTICS_EXPNAME}" \
  $(printf '%q' "${OPTICS_LOGDIR}") \
  $(printf '%q' "${OPTICS_TENSORBOARD_DIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

run_training_then_eval \
  "narrow_lens" \
  "1" \
  "${NARROW_EXPNAME}" \
  $(printf '%q' "${NARROW_LOGDIR}") \
  $(printf '%q' "${NARROW_TENSORBOARD_DIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[15,15]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

echo "[gpu1_pipeline] finished"
exec bash
EOF
}

build_gpu0_eval_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU0_LOG_FILE}")) 2>&1
$(build_pipeline_common)

echo "[gpu0_eval] running Base eye + Multi-eye manual eval"

run_eval_only \
  "base_eye" \
  "0" \
  "${BASE_EXPNAME}" \
  $(printf '%q' "${BASE_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

run_eval_only \
  "multi_eye" \
  "0" \
  "${MULTI_EXPNAME}" \
  $(printf '%q' "${MULTI_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.num_eyes=[1,3]" \
  "env.agents.agent.eyes.eye.lon_range=[-30,30]" \
  "env.agents.agent.eyes.eye.lat_range=[-5,5]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

echo "[gpu0_eval] finished"
exec bash
EOF
}

build_gpu1_eval_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU1_LOG_FILE}")) 2>&1
$(build_pipeline_common)

echo "[gpu1_eval] running Optics eye + Narrow lens manual eval"

run_eval_only \
  "optics_eye" \
  "1" \
  "${OPTICS_EXPNAME}" \
  $(printf '%q' "${OPTICS_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

run_eval_only \
  "narrow_lens" \
  "1" \
  "${NARROW_EXPNAME}" \
  $(printf '%q' "${NARROW_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[15,15]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

echo "[gpu1_eval] finished"
exec bash
EOF
}

build_gpu0_scene_eval_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU0_LOG_FILE}")) 2>&1
$(build_pipeline_common)

echo "[gpu0_scene_eval] running Base eye + Multi-eye manual scene eval"

run_scene_eval_only \
  "base_eye" \
  "0" \
  "${BASE_EXPNAME}" \
  $(printf '%q' "${BASE_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

run_scene_eval_only \
  "multi_eye" \
  "0" \
  "${MULTI_EXPNAME}" \
  $(printf '%q' "${MULTI_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.num_eyes=[1,3]" \
  "env.agents.agent.eyes.eye.lon_range=[-30,30]" \
  "env.agents.agent.eyes.eye.lat_range=[-5,5]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

echo "[gpu0_scene_eval] finished"
exec bash
EOF
}

build_gpu1_scene_eval_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU1_LOG_FILE}")) 2>&1
$(build_pipeline_common)

echo "[gpu1_scene_eval] running Optics eye + Narrow lens manual scene eval"

run_scene_eval_only \
  "optics_eye" \
  "1" \
  "${OPTICS_EXPNAME}" \
  $(printf '%q' "${OPTICS_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

run_scene_eval_only \
  "narrow_lens" \
  "1" \
  "${NARROW_EXPNAME}" \
  $(printf '%q' "${NARROW_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[15,15]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

echo "[gpu1_scene_eval] finished"
exec bash
EOF
}

build_render_common() {
  cat <<EOF
render_lineage() {
  local label="\$1"
  local gpu_id="\$2"
  local expname="\$3"
  local logdir="\$4"
  shift 4
  local overrides=("\$@")

  local checkpoint_glob="\${logdir}/${CHECKPOINT_DIRNAME}/${CHECKPOINT_PREFIX}_*.zip"
  local output_name="checkpoint_lineage_\${RENDER_LAYOUT}"
  local explicit_checkpoints=()

  if [ -f "\${logdir}/best_model.zip" ]; then
    explicit_checkpoints+=("\${logdir}/best_model.zip")
  fi
  if [ -f "\${logdir}/policy.pt" ]; then
    explicit_checkpoints+=("\${logdir}/policy.pt")
  fi

  echo ""
  echo "===== [\${label}] checkpoint render start ====="
  echo "[\${label}] checkpoint_glob=\${checkpoint_glob}"

  local render_cmd=(
    env
    CUDA_VISIBLE_DEVICES="\${gpu_id}"
    bash
    $(printf '%q' "${RUNNER_SCRIPT}")
    tools/render_evolution.py
    --checkpoint-glob
    "\${checkpoint_glob}"
    --output-name
    "\${output_name}"
    --fps
    "${RENDER_FPS}"
    --layout
    "${RENDER_LAYOUT}"
    task=tracking
    env/renderer=${EXPERIMENT_RENDERER}
    record_source=agent_vision
    env.add_overlays=false
    env.debug_overlays_size=0
    expname="\${expname}"
    "logdir=\${logdir}"
    +eval_env.renderer.save_mode=${RENDER_SAVE_MODE}
  )
  if [ "\${#explicit_checkpoints[@]}" -gt 0 ]; then
    render_cmd+=(--checkpoints)
    render_cmd+=("\${explicit_checkpoints[@]}")
  fi
  render_cmd+=("\${overrides[@]}")

  echo "[\${label}] command: \${render_cmd[*]}"
  "\${render_cmd[@]}"
  echo "===== [\${label}] checkpoint render done ====="
}
EOF
}

build_gpu0_render_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU0_LOG_FILE}")) 2>&1
$(build_render_common)

echo "[gpu0_render] rendering Base eye + Multi-eye checkpoint lineage"

render_lineage \
  "base_eye" \
  "0" \
  "${BASE_EXPNAME}" \
  $(printf '%q' "${BASE_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

render_lineage \
  "multi_eye" \
  "0" \
  "${MULTI_EXPNAME}" \
  $(printf '%q' "${MULTI_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.num_eyes=[1,3]" \
  "env.agents.agent.eyes.eye.lon_range=[-30,30]" \
  "env.agents.agent.eyes.eye.lat_range=[-5,5]" \
  "env.agents.agent.eyes.eye.fov=[45,45]"

echo "[gpu0_render] finished"
exec bash
EOF
}

build_gpu1_render_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU1_LOG_FILE}")) 2>&1
$(build_render_common)

echo "[gpu1_render] rendering Optics eye + Narrow lens checkpoint lineage"

render_lineage \
  "optics_eye" \
  "1" \
  "${OPTICS_EXPNAME}" \
  $(printf '%q' "${OPTICS_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

render_lineage \
  "narrow_lens" \
  "1" \
  "${NARROW_EXPNAME}" \
  $(printf '%q' "${NARROW_LOGDIR}") \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[15,15]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75"

echo "[gpu1_render] finished"
exec bash
EOF
}

tmux_window() {
  local window_name="$1"
  local script_text="$2"
  tmux new-window -t "${SESSION_NAME}" -n "${window_name}" \
    "bash -lc $(printf '%q' "${script_text}")"
}

start_session() {
  if session_exists; then
    echo "tmux session '${SESSION_NAME}' already exists. Use 'attach', 'status', or 'stop'." >&2
    exit 1
  fi

  write_current_run_state

  local control_script
  control_script="$(build_control_script)"
  tmux new-session -d -s "${SESSION_NAME}" -n control \
    "bash -lc $(printf '%q' "${control_script}")"
  tmux_window gpu0_pipeline "$(build_gpu0_pipeline_script)"
  tmux_window gpu1_pipeline "$(build_gpu1_pipeline_script)"
  tmux select-window -t "${SESSION_NAME}:control"

  echo "Started tmux session '${SESSION_NAME}'."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
}

run_eval_window() {
  local mode="$1"
  if ! session_exists; then
    echo "tmux session '${SESSION_NAME}' does not exist. Start it first." >&2
    exit 1
  fi

  local suffix
  suffix="$(date +%H%M%S)"
  case "${mode}" in
    gpu0)
      tmux_window "gpu0_eval_${suffix}" "$(build_gpu0_eval_script)"
      ;;
    gpu1)
      tmux_window "gpu1_eval_${suffix}" "$(build_gpu1_eval_script)"
      ;;
    all)
      tmux_window "gpu0_eval_${suffix}" "$(build_gpu0_eval_script)"
      tmux_window "gpu1_eval_${suffix}" "$(build_gpu1_eval_script)"
      ;;
    *)
      echo "Unknown eval mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

run_scene_eval_window() {
  local mode="$1"
  if ! session_exists; then
    echo "tmux session '${SESSION_NAME}' does not exist. Start it first." >&2
    exit 1
  fi

  local suffix
  suffix="$(date +%H%M%S)"
  case "${mode}" in
    gpu0)
      tmux_window "gpu0_scene_eval_${suffix}" "$(build_gpu0_scene_eval_script)"
      ;;
    gpu1)
      tmux_window "gpu1_scene_eval_${suffix}" "$(build_gpu1_scene_eval_script)"
      ;;
    all)
      tmux_window "gpu0_scene_eval_${suffix}" "$(build_gpu0_scene_eval_script)"
      tmux_window "gpu1_scene_eval_${suffix}" "$(build_gpu1_scene_eval_script)"
      ;;
    *)
      echo "Unknown scene eval mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

run_render_window() {
  local mode="$1"
  if ! session_exists; then
    echo "tmux session '${SESSION_NAME}' does not exist. Start it first." >&2
    exit 1
  fi

  local suffix
  suffix="$(date +%H%M%S)"
  case "${mode}" in
    gpu0)
      tmux_window "gpu0_render_${suffix}" "$(build_gpu0_render_script)"
      ;;
    gpu1)
      tmux_window "gpu1_render_${suffix}" "$(build_gpu1_render_script)"
      ;;
    all)
      tmux_window "gpu0_render_${suffix}" "$(build_gpu0_render_script)"
      tmux_window "gpu1_render_${suffix}" "$(build_gpu1_render_script)"
      ;;
    *)
      echo "Unknown render mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

show_status() {
  if ! session_exists; then
    echo "tmux session '${SESSION_NAME}' is not running."
    return 0
  fi
  tmux list-windows -t "${SESSION_NAME}"
  if [ -f "${CURRENT_RUN_FILE}" ]; then
    echo "Current run metadata: ${CURRENT_RUN_FILE}"
  fi
}

stop_session() {
  if session_exists; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "Stopped tmux session '${SESSION_NAME}'."
  else
    echo "tmux session '${SESSION_NAME}' is not running."
  fi
}

attach_session() {
  if ! session_exists; then
    echo "tmux session '${SESSION_NAME}' does not exist." >&2
    exit 1
  fi
  exec tmux attach-session -t "${SESSION_NAME}"
}

tail_logs() {
  load_current_run_state
  echo "Tailing logs for run ${RUN_TIMESTAMP}"
  echo "  GPU 0: ${GPU0_LOG_FILE}"
  echo "  GPU 1: ${GPU1_LOG_FILE}"
  exec tail -F "${GPU0_LOG_FILE}" "${GPU1_LOG_FILE}"
}

main() {
  local command="${1:-up}"
  case "${command}" in
    up|start)
      require_tmux
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      start_session
      ;;
    attach)
      require_tmux
      attach_session
      ;;
    status)
      require_tmux
      show_status
      ;;
    tail)
      tail_logs
      ;;
    stop|down)
      require_tmux
      stop_session
      ;;
    eval-gpu0)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_eval_window gpu0
      ;;
    eval-gpu1)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_eval_window gpu1
      ;;
    eval-all)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_eval_window all
      ;;
    scene-eval-gpu0)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_scene_eval_window gpu0
      ;;
    scene-eval-gpu1)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_scene_eval_window gpu1
      ;;
    scene-eval-all)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_scene_eval_window all
      ;;
    render-gpu0)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_render_window gpu0
      ;;
    render-gpu1)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_render_window gpu1
      ;;
    render-all)
      require_tmux
      load_current_run_state
      require_runner_script
      warn_wandb_if_requested
      PYTHON_BIN="$(resolve_python_bin)"
      ensure_dirs
      run_render_window all
      ;;
    *)
      echo "Usage: $0 [up|attach|status|tail|stop|eval-gpu0|eval-gpu1|eval-all|scene-eval-gpu0|scene-eval-gpu1|scene-eval-all|render-gpu0|render-gpu1|render-all]" >&2
      exit 1
      ;;
  esac
}

main "$@"
