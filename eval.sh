ROOT=/home/sumingyu/TRAPO
DATA=$ROOT/$YOUR_DATA_PATH
# DATA=$ROOT/data/valid.minerva.parquet

# Math 7B Base
# MODEL_NAME=Qwen2.5-Math-7B-Base
# MODEL_PATH=/data/public_checkpoints/Qwen2.5-Math-7B
# TEMPLATE=simplerl
# ADD_OAT_EVALUATE=Fasle
# NO_SPLIT_THINK=True

# SFT Model
# MODEL_NAME=sft
# MODEL_PATH=/data1/sumingyu/checkpoint/global_step5672_hf/global_step5672_hf
# TEMPLATE=simplerl
# ADD_OAT_EVALUATE=True
# NO_SPLIT_THINK=True

# MODEL_NAME=LUFFY
# MODEL_PATH=/data1/sumingyu/checkpoint/LUFFY-Qwen-Math-7B-Zero
# TEMPLATE=own
# ADD_OAT_EVALUATE=False
# NO_SPLIT_THINK=False

# math-instruct
MODEL_NAME=Qwen2.5-Math-7B-Instruct
MODEL_PATH=/data/public_checkpoints/Qwen2.5-Math-7B-Instruct
TEMPLATE=simplerl
ADD_OAT_EVALUATE=False
NO_SPLIT_THINK=True


OUTPUT_DIR=./results_$MODEL_NAME/
mkdir -p $OUTPUT_DIR


# if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
#   TEMPLATE=prime
# elif [ $MODEL_NAME == "simple-rl-zero" ]; then
#   TEMPLATE=qwen
# else
#   TEMPLATE=simplerl
# fi

CUDA_VISIBLE_DEVICES=0,3,4,5 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate $ADD_OAT_EVALUATE \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --no_split_think $NO_SPLIT_THINK \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log