START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
SAVE_ROOT=${SAVE_ROOT:-visualization/}

mkdir -p $SAVE_ROOT

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name $CONFIG_NAME \
    --port $START_PORT \
    --save_root $SAVE_ROOT

