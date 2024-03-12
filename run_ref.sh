NUM_LAYERS=50
DATASET="CIFAR100"
MODEL_PATH=resnet"$NUM_LAYERS"_"$DATASET"

python resnet_ref.py --num_layers $NUM_LAYERS --dataset $DATASET | \
tee $MODEL_PATH/$MODEL_PATH.log --append
