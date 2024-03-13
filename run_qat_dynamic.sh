NUM_LAYERS=50
# DATASET="CIFAR100"
DATASET="ImageNet"
# QUAN="dynamic"
QUAN="static"
# QUAN="qat"
# QUAN='fp32'

MODEL_PATH=resnet"$NUM_LAYERS"_"$DATASET"_"$QUAN"

echo $MODEL_PATH
mkdir -p $MODEL_PATH
python resnet_ref.py \
--arch $NUM_LAYERS \
--dataset $DATASET \
--verbose True \
--batch 128 \
--only_eval True \
--quan $QUAN \
| tee $MODEL_PATH/$MODEL_PATH.log --append
