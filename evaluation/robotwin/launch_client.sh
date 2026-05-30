#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

save_root=${1:-'./results'}
task_name=${2:-"adjust_bottle"}

policy_name=ACT
task_config=${TASK_CONFIG:-demo_clean}
train_config_name=0
model_name=0
seed=${SEED:-0}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-10}
ROLLOUT_LOG_DIR=${ROLLOUT_LOG_DIR:-${save_root}/rollouts}
PROMPT_INDEX=${PROMPT_INDEX:-}
SAMPLING_SEED=${SAMPLING_SEED:-}
SAMPLING_SEED_PER_ENV=${SAMPLING_SEED_PER_ENV:-}
POLICY_CHECKPOINT=${POLICY_CHECKPOINT:-}
REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT:-}
RUN_ID=${RUN_ID:-}

extra_args=()
[ -n "${RUN_ID}" ] && extra_args+=(--run_id "${RUN_ID}")
[ -n "${ACTION_NUM_INFERENCE_STEPS:-}" ] && extra_args+=(--action_num_inference_steps "${ACTION_NUM_INFERENCE_STEPS}")
[ -n "${PROMPT_INDEX}" ] && extra_args+=(--prompt_index "${PROMPT_INDEX}")
[ -n "${SAMPLING_SEED}" ] && extra_args+=(--sampling_seed "${SAMPLING_SEED}")
[ -n "${SAMPLING_SEED_PER_ENV}" ] && extra_args+=(--sampling_seed_per_env "${SAMPLING_SEED_PER_ENV}")
[ -n "${POLICY_CHECKPOINT}" ] && extra_args+=(--policy_checkpoint "${POLICY_CHECKPOINT}")
[ -n "${REFERENCE_CHECKPOINT}" ] && extra_args+=(--reference_checkpoint "${REFERENCE_CHECKPOINT}")

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --server_host ${SERVER_HOST} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num ${TEST_NUM} \
    --port ${PORT} \
    --rollout_log_dir ${ROLLOUT_LOG_DIR} \
    "${extra_args[@]}"
