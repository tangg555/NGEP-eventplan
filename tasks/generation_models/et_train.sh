#!/bin/bash
set -e

CURRENT_DIR=$(dirname $(readlink -f "$0"))
BASE_DIR=$(dirname $(dirname ${CURRENT_DIR}))

EXPERIMENT_GROUP=generation_models
OUTPUT_DIR=${BASE_DIR}/output/${EXPERIMENT_GROUP}
DATASETS_DIR=${BASE_DIR}/datasets/${EXPERIMENT_GROUP}
RESOURCES_DIR=${BASE_DIR}/resources

# reference: https://bummingboy.top/2017/12/19/shell
# %20-%20%E5%8F%82%E6%95%B0%E8%A7%A3%E6%9E%90%E4%B8%89%E7%A7%8D%E6%96%B9%E5%BC%8F
# (%E6%89%8B%E5%B7%A5,%20getopts,%20getopt)/

ARGS=`getopt -a -o t:d:m:b:h -l task_name:,data_name:,model_name:,batch_size:,help -- "$@"`
function usage() {
    echo  'help'
}
[ $? -ne 0 ] && usage
#set -- "${ARGS}"
eval set -- "${ARGS}"
while true
do
      case "$1" in
      -t|--task_name)
              task_name="$2"
              shift
              ;;
      -d|--data_name)
              data_name="$2"
              shift
              ;;
      -m|--model_name)
              model_name="$2"
              shift
              ;;
      -b|--batch_size)
              batch_size="$2"
              shift
              ;;
      -h|--help)
              usage
              ;;
      --)
              shift
              break
              ;;
      esac
shift
done
echo task_name:$task_name data_name:$data_name model_name:$model_name batch_size:$batch_size

# e.g. bart_train
TASK_NAME=${task_name}
# e.g. roc-stories
DATA_NAME=${data_name}
# e.g. event-bart
MODEL_NAME=${model_name}
# e.g. 64
BATCH_SIZE=${batch_size}
EXPERIMENT_NAME=${MODEL_NAME}-${DATA_NAME}

# GPU Memory size is 12GB
python ${BASE_DIR}/tasks/${EXPERIMENT_GROUP}/${TASK_NAME}.py --data_dir=${DATASETS_DIR}/${DATA_NAME} \
    --experiment_name=${EXPERIMENT_NAME} \
    --model_name_or_path=${RESOURCES_DIR}/external_models/bart-base \
    --output_dir=${OUTPUT_DIR} \
    --model_name=${MODEL_NAME} \
    --train_batch_size=${BATCH_SIZE} \
    --eval_batch_size=${BATCH_SIZE}  \
    --learning_rate=3e-6