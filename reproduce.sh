#!/bin/bash


DEVICE_ID=0,1,2,3
TTA=4

MODEL_NAMES=(
  model_a_00
  model_a_02
  model_b_00
  model_b_01
  model_b_02
  model_b_03
  model_b_04
  model_b_05
  model_b_06
  model_b_07
  model_b_08
  )


##################################################################################
# inference all models
##################################################################################
for MODEL_NAME in ${MODEL_NAMES[@]}; do
  CONFIG=configs/$MODEL_NAME.yaml
  CHECKPOINT=checkpoints/$MODEL_NAME.pth
  OUTPUT_PATH=inference_results/$MODEL_NAME

  CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py inference with $CONFIG evaluation.batch_size=2 inference.output_path=$OUTPUT_PATH transform.params.tta=$TTA inference.split=dev checkpoint=$CHECKPOINT -f
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py inference with $CONFIG evaluation.batch_size=2 inference.output_path=$OUTPUT_PATH transform.params.tta=$TTA inference.split=test_dev checkpoint=$CHECKPOINT -f
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py inference with $CONFIG evaluation.batch_size=2 inference.output_path=$OUTPUT_PATH transform.params.tta=$TTA inference.split=test checkpoint=$CHECKPOINT -f
done

# The result of model_a_01 was generated with no tta because of just my fault.
TTA=1
MODEL_NAME=model_a_01
CONFIG=configs/$MODEL_NAME.yaml
CHECKPOINT=checkpoints/$MODEL_NAME.pth
OUTPUT_PATH=inference_results/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py inference with $CONFIG evaluation.batch_size=2 inference.output_path=$OUTPUT_PATH transform.params.tta=$TTA inference.split=dev checkpoint=$CHECKPOINT -f
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py inference with $CONFIG evaluation.batch_size=2 inference.output_path=$OUTPUT_PATH transform.params.tta=$TTA inference.split=test_dev checkpoint=$CHECKPOINT -f
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py inference with $CONFIG evaluation.batch_size=2 inference.output_path=$OUTPUT_PATH transform.params.tta=$TTA inference.split=test checkpoint=$CHECKPOINT -f

##################################################################################
# evaluate & inference 
##################################################################################
MODEL_NAMES=(
  model_a_00
  model_a_01
  model_a_02
  model_b_00
  model_b_01
  model_b_02
  model_b_03
  model_b_04
  model_b_05
  model_b_06
  model_b_07
  model_b_08
  )

OUTPUTS=
for MODEL_NAME in ${MODEL_NAMES[@]}; do
  OUTPUT_PATH=inference_results/$MODEL_NAME
  OUTPUTS=$OUTPUTS,$OUTPUT_PATH
done

OUTPUTS=${OUTPUTS:1}

python tools/evaluate.py --input_dir $OUTPUTS
python tools/make_submission.py --input_dir $OUTPUTS --output=submissions/reproduce.csv
