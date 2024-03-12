NUM_LAYERS=50
# DATASET="CIFAR100"
DATASET="ImageNet"
MODEL_PATH=resnet"$NUM_LAYERS"_"$DATASET"
QAT=True

if [ "$QAT" = True ]; then
    MODEL_PATH=resnet"$NUM_LAYERS"_"$DATASET"_QAT
    echo $MODEL_PATH
    mkdir -p $MODEL_PATH
    python resnet_ref.py \
    --num_layers $NUM_LAYERS \
    --dataset $DATASET \
    --verbose True \
    --batch_size 128 \
    --qat $QAT \
    | tee $MODEL_PATH/$MODEL_PATH.log --append
else
    MODEL_PATH=resnet"$NUM_LAYERS"_"$DATASET"
    echo $MODEL_PATH
    mkdir -p $MODEL_PATH
    python resnet_ref.py \
    --num_layers $NUM_LAYERS \
    --dataset $DATASET \
    --verbose True \
    --batch_size 128 \
    | tee $MODEL_PATH/$MODEL_PATH.log --append
fi

# --only_eval True \