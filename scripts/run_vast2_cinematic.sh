#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-evo_cinematic}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/vast2}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/logs/vast2_runs}"
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
EVAL_FREQ="${EVAL_FREQ:-10000}"

DETECTION_EXPNAME="${DETECTION_EXPNAME:-exp_detection_deep_sea_vast2}"
NAVIGATION_EXPNAME="${NAVIGATION_EXPNAME:-exp_navigation_cambrian_shallows_vast2}"
DETECTION_LOGDIR="${DETECTION_LOGDIR:-${LOG_ROOT}/${DETECTION_EXPNAME}}"
NAVIGATION_LOGDIR="${NAVIGATION_LOGDIR:-${LOG_ROOT}/${NAVIGATION_EXPNAME}}"

DETECTION_TIMESTEPS="${DETECTION_TIMESTEPS:-${TOTAL_TIMESTEPS}}"
NAVIGATION_TIMESTEPS="${NAVIGATION_TIMESTEPS:-${TOTAL_TIMESTEPS}}"
DETECTION_EVAL_EPISODES="${DETECTION_EVAL_EPISODES:-1}"
NAVIGATION_EVAL_EPISODES="${NAVIGATION_EVAL_EPISODES:-1}"
EVAL_SAVE_MODE="${EVAL_SAVE_MODE:-WEBP}"
USE_TENSORBOARD="${USE_TENSORBOARD:-0}"
USE_WANDB="${USE_WANDB:-0}"

GPU0_LOG_FILE="${GPU0_LOG_FILE:-${RUN_LOG_DIR}/gpu0_detection.log}"
GPU1_LOG_FILE="${GPU1_LOG_FILE:-${RUN_LOG_DIR}/gpu1_navigation.log}"
DETECTION_TENSORBOARD_DIR="${DETECTION_TENSORBOARD_DIR:-${RUN_LOG_DIR}/tensorboard/detection}"
NAVIGATION_TENSORBOARD_DIR="${NAVIGATION_TENSORBOARD_DIR:-${RUN_LOG_DIR}/tensorboard/navigation}"

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
    "${DETECTION_LOGDIR}" \
    "${NAVIGATION_LOGDIR}" \
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
cd $(printf '%q' "${REPO_ROOT}")
mkdir -p "\$MPLCONFIGDIR" "\$XDG_CACHE_HOME" "\$MESA_SHADER_CACHE_DIR" \
  $(printf '%q' "${RUN_LOG_DIR}") \
  $(printf '%q' "${DETECTION_LOGDIR}") \
  $(printf '%q' "${NAVIGATION_LOGDIR}") \
  $(printf '%q' "${DETECTION_TENSORBOARD_DIR}") \
  $(printf '%q' "${NAVIGATION_TENSORBOARD_DIR}")
EOF
}

write_current_run_state() {
  mkdir -p "${STATE_DIR}"
  cat > "${CURRENT_RUN_FILE}" <<EOF
RUN_TIMESTAMP=$(printf '%q' "${RUN_TIMESTAMP}")
RUN_LOG_DIR=$(printf '%q' "${RUN_LOG_DIR}")
GPU0_LOG_FILE=$(printf '%q' "${GPU0_LOG_FILE}")
GPU1_LOG_FILE=$(printf '%q' "${GPU1_LOG_FILE}")
DETECTION_LOGDIR=$(printf '%q' "${DETECTION_LOGDIR}")
NAVIGATION_LOGDIR=$(printf '%q' "${NAVIGATION_LOGDIR}")
DETECTION_EXPNAME=$(printf '%q' "${DETECTION_EXPNAME}")
NAVIGATION_EXPNAME=$(printf '%q' "${NAVIGATION_EXPNAME}")
TOTAL_TIMESTEPS=$(printf '%q' "${TOTAL_TIMESTEPS}")
EVAL_FREQ=$(printf '%q' "${EVAL_FREQ}")
EVAL_SAVE_MODE=$(printf '%q' "${EVAL_SAVE_MODE}")
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
  - det_train: GPU 0, detection + deep_sea training, then agent_vision eval
  - nav_train: GPU 1, navigation + cambrian_shallows training, then agent_vision eval

Useful commands from another shell:
  ./scripts/run_vast2_cinematic.sh attach
  ./scripts/run_vast2_cinematic.sh status
  ./scripts/run_vast2_cinematic.sh tail
  ./scripts/run_vast2_cinematic.sh eval-det
  ./scripts/run_vast2_cinematic.sh eval-nav
  ./scripts/run_vast2_cinematic.sh eval-all
  ./scripts/run_vast2_cinematic.sh stop

Outputs:
  Detection : ${DETECTION_LOGDIR}
  Navigation: ${NAVIGATION_LOGDIR}
Train logs:
  GPU 0: ${GPU0_LOG_FILE}
  GPU 1: ${GPU1_LOG_FILE}
Timesteps per task: ${TOTAL_TIMESTEPS}
Eval frequency: ${EVAL_FREQ}
Eval save mode: ${EVAL_SAVE_MODE}
TensorBoard: ${USE_TENSORBOARD} (dirs under ${RUN_LOG_DIR}/tensorboard)
WandB: ${USE_WANDB} (not wired in this repo)
MSG
exec bash
EOF
}

build_detection_train_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU0_LOG_FILE}")) 2>&1
echo "[det_train] starting training on GPU 0"
echo "[det_train] run_timestamp=${RUN_TIMESTAMP}"
echo "[det_train] train_log=${GPU0_LOG_FILE}"
echo "[det_train] total_timesteps=${DETECTION_TIMESTEPS} eval_freq=${EVAL_FREQ}"
if [ $(printf '%q' "${USE_WANDB}") = "1" ]; then
  echo "[det_train] USE_WANDB=1 requested but WandB integration is not available in this repo." >&2
fi

train_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0
  $(printf '%q' "${PYTHON_BIN}")
  cambrian/main.py
  --train
  example=detection
  env/renderer=deep_sea
  expname=${DETECTION_EXPNAME}
  logdir=$(printf '%q' "${DETECTION_LOGDIR}")
  trainer.total_timesteps=${DETECTION_TIMESTEPS}
  trainer.callbacks.eval_callback.eval_freq=${EVAL_FREQ}
)

if [ $(printf '%q' "${USE_TENSORBOARD}") = "1" ]; then
  train_cmd+=(+trainer.model.tensorboard_log=$(printf '%q' "${DETECTION_TENSORBOARD_DIR}"))
fi

echo "[det_train] command: \${train_cmd[*]}"
"\${train_cmd[@]}"

if [ -f $(printf '%q' "${DETECTION_LOGDIR}")/best_model.zip ]; then
  echo "[det_train] training complete, running agent_vision eval"
  eval_cmd=(
    env
    CUDA_VISIBLE_DEVICES=0
    $(printf '%q' "${PYTHON_BIN}")
    cambrian/main.py
    --eval
    example=detection
    env/renderer=deep_sea
    record_source=agent_vision
    trainer/model=loaded_model
    expname=${DETECTION_EXPNAME}
    logdir=$(printf '%q' "${DETECTION_LOGDIR}")
    eval_env.n_eval_episodes=${DETECTION_EVAL_EPISODES}
    eval_env.save_filename=eval_agent_vision
    eval_env.renderer.save_mode=${EVAL_SAVE_MODE}
  )
  echo "[det_train] command: \${eval_cmd[*]}"
  "\${eval_cmd[@]}"
else
  echo "[det_train] best_model.zip not found, skipping automatic eval" >&2
fi

echo "[det_train] finished"
exec bash
EOF
}

build_navigation_train_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU1_LOG_FILE}")) 2>&1
echo "[nav_train] starting training on GPU 1"
echo "[nav_train] run_timestamp=${RUN_TIMESTAMP}"
echo "[nav_train] train_log=${GPU1_LOG_FILE}"
echo "[nav_train] total_timesteps=${NAVIGATION_TIMESTEPS} eval_freq=${EVAL_FREQ}"
if [ $(printf '%q' "${USE_WANDB}") = "1" ]; then
  echo "[nav_train] USE_WANDB=1 requested but WandB integration is not available in this repo." >&2
fi

train_cmd=(
  env
  CUDA_VISIBLE_DEVICES=1
  $(printf '%q' "${PYTHON_BIN}")
  cambrian/main.py
  --train
  example=navigation
  env/renderer=cambrian_shallows
  expname=${NAVIGATION_EXPNAME}
  logdir=$(printf '%q' "${NAVIGATION_LOGDIR}")
  trainer.total_timesteps=${NAVIGATION_TIMESTEPS}
  trainer.callbacks.eval_callback.eval_freq=${EVAL_FREQ}
)

if [ $(printf '%q' "${USE_TENSORBOARD}") = "1" ]; then
  train_cmd+=(+trainer.model.tensorboard_log=$(printf '%q' "${NAVIGATION_TENSORBOARD_DIR}"))
fi

echo "[nav_train] command: \${train_cmd[*]}"
"\${train_cmd[@]}"

if [ -f $(printf '%q' "${NAVIGATION_LOGDIR}")/best_model.zip ]; then
  echo "[nav_train] training complete, running agent_vision eval"
  eval_cmd=(
    env
    CUDA_VISIBLE_DEVICES=1
    $(printf '%q' "${PYTHON_BIN}")
    cambrian/main.py
    --eval
    example=navigation
    env/renderer=cambrian_shallows
    record_source=agent_vision
    trainer/model=loaded_model
    expname=${NAVIGATION_EXPNAME}
    logdir=$(printf '%q' "${NAVIGATION_LOGDIR}")
    eval_env.n_eval_episodes=${NAVIGATION_EVAL_EPISODES}
    eval_env.save_filename=eval_agent_vision
    eval_env.renderer.save_mode=${EVAL_SAVE_MODE}
  )
  echo "[nav_train] command: \${eval_cmd[*]}"
  "\${eval_cmd[@]}"
else
  echo "[nav_train] best_model.zip not found, skipping automatic eval" >&2
fi

echo "[nav_train] finished"
exec bash
EOF
}

build_detection_eval_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU0_LOG_FILE}")) 2>&1
if [ ! -f $(printf '%q' "${DETECTION_LOGDIR}")/best_model.zip ]; then
  echo "[det_eval] best_model.zip not found at ${DETECTION_LOGDIR}" >&2
  exec bash
fi
echo "[det_eval] evaluating latest detection checkpoint with agent_vision"
eval_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0
  $(printf '%q' "${PYTHON_BIN}")
  cambrian/main.py
  --eval
  example=detection
  env/renderer=deep_sea
  record_source=agent_vision
  trainer/model=loaded_model
  expname=${DETECTION_EXPNAME}
  logdir=$(printf '%q' "${DETECTION_LOGDIR}")
  eval_env.n_eval_episodes=${DETECTION_EVAL_EPISODES}
  eval_env.save_filename=eval_agent_vision_latest
  eval_env.renderer.save_mode=${EVAL_SAVE_MODE}
)
echo "[det_eval] command: \${eval_cmd[*]}"
"\${eval_cmd[@]}"
echo "[det_eval] finished"
exec bash
EOF
}

build_navigation_eval_script() {
  cat <<EOF
set -euo pipefail
$(common_exports)
exec > >(tee -a $(printf '%q' "${GPU1_LOG_FILE}")) 2>&1
if [ ! -f $(printf '%q' "${NAVIGATION_LOGDIR}")/best_model.zip ]; then
  echo "[nav_eval] best_model.zip not found at ${NAVIGATION_LOGDIR}" >&2
  exec bash
fi
echo "[nav_eval] evaluating latest navigation checkpoint with agent_vision"
eval_cmd=(
  env
  CUDA_VISIBLE_DEVICES=1
  $(printf '%q' "${PYTHON_BIN}")
  cambrian/main.py
  --eval
  example=navigation
  env/renderer=cambrian_shallows
  record_source=agent_vision
  trainer/model=loaded_model
  expname=${NAVIGATION_EXPNAME}
  logdir=$(printf '%q' "${NAVIGATION_LOGDIR}")
  eval_env.n_eval_episodes=${NAVIGATION_EVAL_EPISODES}
  eval_env.save_filename=eval_agent_vision_latest
  eval_env.renderer.save_mode=${EVAL_SAVE_MODE}
)
echo "[nav_eval] command: \${eval_cmd[*]}"
"\${eval_cmd[@]}"
echo "[nav_eval] finished"
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
  tmux_window det_train "$(build_detection_train_script)"
  tmux_window nav_train "$(build_navigation_train_script)"
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
    det)
      tmux_window "det_eval_${suffix}" "$(build_detection_eval_script)"
      ;;
    nav)
      tmux_window "nav_eval_${suffix}" "$(build_navigation_eval_script)"
      ;;
    all)
      tmux_window "det_eval_${suffix}" "$(build_detection_eval_script)"
      tmux_window "nav_eval_${suffix}" "$(build_navigation_eval_script)"
      ;;
    *)
      echo "Unknown eval mode: ${mode}" >&2
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
      warn_wandb_if_requested
      ensure_dirs
      PYTHON_BIN="$(resolve_python_bin)"
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
    eval-det)
      require_tmux
      load_current_run_state
      warn_wandb_if_requested
      ensure_dirs
      PYTHON_BIN="$(resolve_python_bin)"
      run_eval_window det
      ;;
    eval-nav)
      require_tmux
      load_current_run_state
      warn_wandb_if_requested
      ensure_dirs
      PYTHON_BIN="$(resolve_python_bin)"
      run_eval_window nav
      ;;
    eval-all)
      require_tmux
      load_current_run_state
      warn_wandb_if_requested
      ensure_dirs
      PYTHON_BIN="$(resolve_python_bin)"
      run_eval_window all
      ;;
    *)
      echo "Usage: $0 [up|attach|status|tail|stop|eval-det|eval-nav|eval-all]" >&2
      exit 1
      ;;
  esac
}

main "$@"
