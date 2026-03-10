#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run.sh}"
PYTHON_BIN="${PYTHON_BIN:-}"

MUJOCO_GL_VALUE="${MUJOCO_GL:-egl}"
MPLCONFIGDIR_VALUE="${MPLCONFIGDIR:-/tmp/mpl}"
JOBLIB_TEMP_FOLDER_VALUE="${JOBLIB_TEMP_FOLDER:-/tmp}"
XDG_CACHE_HOME_VALUE="${XDG_CACHE_HOME:-/tmp/.cache}"
MESA_SHADER_CACHE_DIR_VALUE="${MESA_SHADER_CACHE_DIR:-/tmp/mesa_shader_cache}"

SHOWCASE_MODE="${SHOWCASE_MODE:-showcase}"
SHOWCASE_RENDERER="${SHOWCASE_RENDERER:-cambrian_shallows_showcase}"
SHOWCASE_SCENE_RENDERER="${SHOWCASE_SCENE_RENDERER:-cambrian_shallows_showcase_scene}"
FAITHFUL_RENDERER="${FAITHFUL_RENDERER:-cambrian_shallows}"
VISION_SAVE_MODE="${VISION_SAVE_MODE:-WEBP}"
SIDE_BY_SIDE_SAVE_MODE="${SIDE_BY_SIDE_SAVE_MODE:-WEBP}"
SCENE_SAVE_MODE="${SCENE_SAVE_MODE:-WEBP}"
TRACKING_EVAL_EPISODES="${TRACKING_EVAL_EPISODES:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

usage() {
  cat <<'EOF' >&2
Usage:
  ./scripts/render_cambrian_showcase.sh <variant> <logdir>

Variants:
  base | multi | optics | narrow

Examples:
  ./scripts/render_cambrian_showcase.sh multi logs/vast2_tracking_shallows/vast2_tracking_shallows_multi_eye
  SHOWCASE_MODE=faithful ./scripts/render_cambrian_showcase.sh base logs/vast2_tracking_shallows/vast2_tracking_shallows_base_eye
EOF
  exit 1
}

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

require_runner_script() {
  [ -f "${RUNNER_SCRIPT}" ] || {
    echo "Could not find runner script at ${RUNNER_SCRIPT}" >&2
    exit 1
  }
}

common_env() {
  export MUJOCO_GL="${MUJOCO_GL_VALUE}"
  export MPLCONFIGDIR="${MPLCONFIGDIR_VALUE}"
  export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER_VALUE}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME_VALUE}"
  export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR_VALUE}"
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${MESA_SHADER_CACHE_DIR}"
  cd "${REPO_ROOT}"
}

variant_overrides() {
  local variant="$1"
  case "${variant}" in
    base)
      printf '%s\n' \
        "env/agents@env.agents.agent=point" \
        "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
        "env.agents.agent.eyes.eye.resolution=[20,20]" \
        "env.agents.agent.eyes.eye.fov=[45,45]"
      ;;
    multi)
      printf '%s\n' \
        "env/agents@env.agents.agent=point" \
        "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye" \
        "env.agents.agent.eyes.eye.resolution=[20,20]" \
        "env.agents.agent.eyes.eye.num_eyes=[1,3]" \
        "env.agents.agent.eyes.eye.lon_range=[-30,30]" \
        "env.agents.agent.eyes.eye.lat_range=[-5,5]" \
        "env.agents.agent.eyes.eye.fov=[45,45]"
      ;;
    optics)
      printf '%s\n' \
        "env/agents@env.agents.agent=point" \
        "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
        "env.agents.agent.eyes.eye.resolution=[20,20]" \
        "env.agents.agent.eyes.eye.fov=[45,45]" \
        "env.agents.agent.eyes.eye.aperture.radius=0.75"
      ;;
    narrow)
      printf '%s\n' \
        "env/agents@env.agents.agent=point" \
        "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
        "env.agents.agent.eyes.eye.resolution=[20,20]" \
        "env.agents.agent.eyes.eye.fov=[15,15]" \
        "env.agents.agent.eyes.eye.aperture.radius=0.75"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

showcase_visual_overrides() {
  if [ "${SHOWCASE_MODE}" != "showcase" ]; then
    return 0
  fi
  printf '%s\n' \
    "env.cinematic_mode=true" \
    "env/agents@env.agents.agent=point_cambrian_showcase" \
    "env/agents@env.agents.goal0=tracking_goal_cambrian_showcase" \
    "env/agents@env.agents.adversary0=tracking_adversary_cambrian_showcase"
}

run_eval() {
  local label="$1"
  local logdir="$2"
  local expname="$3"
  local renderer_name="$4"
  local save_filename="$5"
  local save_mode="$6"
  local record_source="${7:-scene}"
  shift 7
  local overrides=("$@")

  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    bash
    "${RUNNER_SCRIPT}"
    cambrian/main.py
    --eval
    task=tracking
    "env/renderer=${renderer_name}"
    trainer/model=loaded_model
    env.add_overlays=false
    env.debug_overlays_size=0
    "expname=${expname}"
    "logdir=${logdir}"
    "eval_env.n_eval_episodes=${TRACKING_EVAL_EPISODES}"
    "eval_env.save_filename=${save_filename}"
    "+eval_env.renderer.save_mode=${save_mode}"
  )
  case "${record_source}" in
    scene)
      ;;
    agent_vision|side_by_side)
      cmd+=("record_source=${record_source}")
      ;;
    *)
      echo "Unknown record_source: ${record_source}. Expected scene|agent_vision|side_by_side." >&2
      exit 1
      ;;
  esac
  cmd+=("${overrides[@]}")
  echo "[${label}] ${cmd[*]}"
  "${cmd[@]}"
}

main() {
  local variant="${1:-}"
  local logdir="${2:-}"
  [ -n "${variant}" ] || usage
  [ -n "${logdir}" ] || usage

  PYTHON_BIN="$(resolve_python_bin)"
  require_runner_script
  common_env

  if [ ! -d "${logdir}" ]; then
    echo "Logdir not found: ${logdir}" >&2
    exit 1
  fi
  if [ ! -f "${logdir}/best_model.zip" ]; then
    echo "best_model.zip not found in ${logdir}" >&2
    exit 1
  fi

  local renderer_name
  local scene_renderer_name
  case "${SHOWCASE_MODE}" in
    faithful)
      renderer_name="${FAITHFUL_RENDERER}"
      scene_renderer_name="${FAITHFUL_RENDERER}"
      ;;
    showcase)
      renderer_name="${SHOWCASE_RENDERER}"
      scene_renderer_name="${SHOWCASE_SCENE_RENDERER}"
      ;;
    *)
      echo "Unknown SHOWCASE_MODE: ${SHOWCASE_MODE}. Expected faithful|showcase." >&2
      exit 1
      ;;
  esac

  local expname
  expname="$(basename "${logdir}")"

  mapfile -t base_overrides < <(variant_overrides "${variant}")
  mapfile -t visual_overrides < <(showcase_visual_overrides)

  local all_overrides=("${base_overrides[@]}" "${visual_overrides[@]}")
  local output_prefix="cambrian_${SHOWCASE_MODE}"

  run_eval "scene" "${logdir}" "${expname}" "${scene_renderer_name}" "${output_prefix}_scene" "${SCENE_SAVE_MODE}" "scene" "${all_overrides[@]}"
  run_eval "agent_vision" "${logdir}" "${expname}" "${renderer_name}" "${output_prefix}_agent_vision" "${VISION_SAVE_MODE}" "agent_vision" "${all_overrides[@]}"
  run_eval "side_by_side" "${logdir}" "${expname}" "${scene_renderer_name}" "${output_prefix}_side_by_side" "${SIDE_BY_SIDE_SAVE_MODE}" "side_by_side" "${all_overrides[@]}"
}

main "$@"
