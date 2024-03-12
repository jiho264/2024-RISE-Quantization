NUM_LAYERS=50
# DATASET="CIFAR100"
DATASET="ImageNet"
MODEL_PATH=resnet"$NUM_LAYERS"_"$DATASET"

python resnet_ref.py \
--num_layers $NUM_LAYERS \
--dataset $DATASET \
--verbose True \
--only_eval True \
--batch_size 256| \
tee $MODEL_PATH/$MODEL_PATH.log --append
