SCHEMES=( "AdaRoundQuantizer" )
BASES=( "AbsMaxQuantizer" "MinMaxQuantizer" "NormQuantizer" "OrgNormQuantizerCode" )
PER_CHANNEL_FLAG="--per_channel"
LR=( "0.01" "0.001")

for _i in "${BASES[@]}"; do
    for _lr in "${LR[@]}"; do
        if [ "$_lr" == "0.01" ]; then
            lr_name="1e-2"
        elif [ "$_lr" == "0.001" ]; then
            lr_name="1e-3"
        fi
        FILENAME="logs/${SCHEMES}_${_i}_CH_W4A32_LR${lr_name}.log"
        echo "Running test case for ${FILENAME}"
        python main_myAdaRound.py --scheme $SCHEMES $PER_CHANNEL_FLAG --lr ${_lr} --BaseScheme $_i | tee ${FILENAME}
        echo ""
        
    done
done