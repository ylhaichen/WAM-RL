#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH


save_root=${1:-'./results'}

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${3:-0}
test_num=${4:-${TEST_NUM:-10}}
start_port=${START_PORT:-29556}
num_gpus=${NUM_GPUS:-4}
rollout_log_dir=${ROLLOUT_LOG_DIR:-${save_root}/rollouts}
client_log_dir=${CLIENT_LOG_DIR:-${save_root}/logs/clients}

task_list_id=${2:-0}

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

if [ -n "${TASK_NAMES:-}" ]; then
  read -r -a task_names <<< "${TASK_NAMES}"
else
  if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
    echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
    exit 1
  fi
  read -r -a task_names <<< "${task_groups[$task_list_id]}"
fi

echo "task_list_id=$task_list_id"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"

mkdir -p "$client_log_dir"

echo -e "\033[32mLaunching ${#task_names[@]} tasks. GPUs assigned by mod ${num_gpus}, ports starting from ${start_port} incrementing.\033[0m"

batch_time=$(date +%Y%m%d_%H%M%S)

pid_file="${client_log_dir}/pids_${batch_time}.txt"
> "$pid_file"
task_file="${client_log_dir}/tasks_${batch_time}.txt"
> "$task_file"

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id=$(( i % num_gpus ))
    port=$(( start_port + gpu_id ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${client_log_dir}/${task_name}_${batch_time}.log"

    echo -e "\033[33m[Task $i] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    extra_args=()
    [ -n "${RUN_ID:-}" ] && extra_args+=(--run_id "$RUN_ID")
    [ -n "${POLICY_CHECKPOINT:-}" ] && extra_args+=(--policy_checkpoint "$POLICY_CHECKPOINT")
    [ -n "${REFERENCE_CHECKPOINT:-}" ] && extra_args+=(--reference_checkpoint "$REFERENCE_CHECKPOINT")
    [ -n "${GROUP_ID:-}" ] && extra_args+=(--group_id "$GROUP_ID")
    [ -n "${GROUP_INDEX:-}" ] && extra_args+=(--group_index "$GROUP_INDEX")
    [ -n "${SAMPLE_IDX:-}" ] && extra_args+=(--sample_idx "$SAMPLE_IDX")
    [ -n "${GROUP_SIZE:-}" ] && extra_args+=(--group_size "$GROUP_SIZE")
    [ -n "${SAMPLING_SEED:-}" ] && extra_args+=(--sampling_seed "$SAMPLING_SEED")
    [ -n "${PROMPT_INDEX:-}" ] && extra_args+=(--prompt_index "$PROMPT_INDEX")
    [ -n "${ACTION_NUM_INFERENCE_STEPS:-}" ] && extra_args+=(--action_num_inference_steps "$ACTION_NUM_INFERENCE_STEPS")

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
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num ${test_num} \
        --port ${port} \
        --rollout_log_dir ${rollout_log_dir} \
        "${extra_args[@]}" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
    echo "${pid} ${task_name} ${log_file}" >> "$task_file"
done

echo -e "\033[32mAll tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"

status=0
while read -r pid task_name log_file; do
    if wait "${pid}"; then
        echo -e "\033[32m[Done] ${task_name}: ${log_file}\033[0m"
    else
        rc=$?
        status=1
        echo -e "\033[31m[Failed] ${task_name}: exit=${rc}, log=${log_file}\033[0m" >&2
        tail -80 "${log_file}" >&2 || true
    fi
done < "$task_file"

if [ "$status" -ne 0 ]; then
    echo "One or more evaluation clients failed." >&2
    exit "$status"
fi

echo -e "\033[32mAll evaluation clients completed successfully.\033[0m"
