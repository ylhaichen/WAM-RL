START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
NUM_GPUS=${NUM_GPUS:-4}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
SAVE_ROOT=${SAVE_ROOT:-./visualization/}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

mkdir -p $SAVE_ROOT

batch_time=$(date +%Y%m%d_%H%M%S)


for ((i=0; i<NUM_GPUS; i++)); do
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Server ${i}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name $CONFIG_NAME \
        --save_root $SAVE_ROOT \
        --port $CURRENT_PORT  > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All ${NUM_GPUS} instances have been launched in the background."
wait
