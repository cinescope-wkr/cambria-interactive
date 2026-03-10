#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SESSION_NAME="${SESSION_NAME:-tracking_eyes_shallows}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/../logs/vast2_tracking_shallows}"
export RUNS_ROOT="${RUNS_ROOT:-${SCRIPT_DIR}/../logs/vast2_tracking_shallows_runs}"

export BASE_EXPNAME="${BASE_EXPNAME:-vast2_tracking_shallows_base_eye}"
export MULTI_EXPNAME="${MULTI_EXPNAME:-vast2_tracking_shallows_multi_eye}"
export OPTICS_EXPNAME="${OPTICS_EXPNAME:-vast2_tracking_shallows_optics_eye}"
export NARROW_EXPNAME="${NARROW_EXPNAME:-vast2_tracking_shallows_narrow_lens}"

export EXPERIMENT_RENDERER="${EXPERIMENT_RENDERER:-cambrian_shallows}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
export EVAL_FREQ="${EVAL_FREQ:-5000}"
export TRACKING_EVAL_EPISODES="${TRACKING_EVAL_EPISODES:-1}"

export EVAL_SAVE_MODE="${EVAL_SAVE_MODE:-WEBP}"
export AUTO_SCENE_EVAL="${AUTO_SCENE_EVAL:-1}"
export SCENE_EVAL_SAVE_MODE="${SCENE_EVAL_SAVE_MODE:-MP4}"

export RENDER_LAYOUT="${RENDER_LAYOUT:-side_by_side}"
export RENDER_SAVE_MODE="${RENDER_SAVE_MODE:-MP4}"
export RENDER_FPS="${RENDER_FPS:-20}"

exec "${SCRIPT_DIR}/run_vast2_tracking_eyes.sh" "$@"
