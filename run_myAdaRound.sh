SCHEMES=( "AbsMaxQuantizer" "MinMaxQuantizer" "NormQuantizer" "OrgNormQuantizerCode" )
DSTDTYPEW=( "INT4" )
DSTDTYPEA=( "INT4" "INT8" "FP32" )
# when using AdaRound, per-ch is default
# when using AdaRound, BN fold is default. 
#   reason is that, activation quantization is apply to the output of the BN layer.
LR=( "0.01" "0.001")

"" > logs/summary_AdaRound.log

for _i in "${SCHEMES[@]}"; do
    for _lr in "${LR[@]}"; do
        if [ "$_lr" == "0.01" ]; then
            lr_name="1e-2"
        elif [ "$_lr" == "0.001" ]; then
            lr_name="1e-3"
        fi
        for _w in "${DSTDTYPEW[@]}"; do
            if [ "$_w" == "INT4" ]; then
                W_bit="4"
            else
                W_bit="8"
            fi
            for _a in "${DSTDTYPEA[@]}"; do
                if [ "$_a" == "FP32" ]; then
                    A_bit="32"
                else
                    if [ "$_a" == "INT4" ]; then
                    A_bit="4"
                    else
                        A_bit="8"
                    fi    
                fi
                FILENAME="logs/${SCHEMES}_${_i}_W${W_bit}A${A_bit}_LR${lr_name}.log"
                echo "Running test case for ${FILENAME}"
                python main_myAdaRound.py --scheme_w $_i --scheme_a $_i --lr ${_lr} --dstDtypeW $_w --dstDtypeA $_a | tee ${FILENAME}
                echo ""
                head -n 2 ${FILENAME} >> logs/summary_AdaRound.log
                tail -n 2 ${FILENAME} >> logs/summary_AdaRound.log
            done
        done
        
    done
done