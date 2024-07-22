# W8A32 or W4A32
SCHEMES=( "AbsMaxQuantizer" "MinMaxQuantizer" "NormQuantizer" "OrgNormQuantizerCode" )
DSTDTYPEW=( "INT4" "INT8" )
FOLDING=( "True" "False" )
PERCHANNEL=( "False" "True" )
L_PNORM=( "2" "2.4")

"" > logs/summary.log

for _i in "${SCHEMES[@]}"; do
    for _j in "${PERCHANNEL[@]}"; do
        if [ "$_j" == "False" ]; then
            PER_Which="Layer"
            PER_CHANNEL_FLAG=""
        else
            PER_Which="CH"
            PER_CHANNEL_FLAG="--per_channel"
        fi
        for _f in "${FOLDING[@]}"; do
            if [ "$_f" == "True" ]; then
                FOLDIED="_folded"
                FOLDING_FLAG="--folding"
            else
                FOLDIED=""
                FOLDING_FLAG=""
            fi
            for _w in "${DSTDTYPEW[@]}"; do
                if [ "$_w" == "INT4" ]; then
                    W_bit="4"
                else
                    W_bit="8"
                fi
                if [ "$_i" == "NormQuantizer" ]; then
                    for _p in "${L_PNORM[@]}"; do
                        if [ "$_p" == "2.4" ]; then
                            p_norm="24"
                        else
                            p_norm="2"
                        fi
                        FILENAME="logs/${_i}_${PER_Which}_W${W_bit}A32_p${p_norm}${FOLDIED}.log"
                        echo "Running test case for ${FILENAME}"
                        # python main_myAdaRound.py --scheme_w $_i $PER_CHANNEL_FLAG $FOLDING_FLAG --dstDtypeW $_w --p ${_p} | tee ${FILENAME}
                        # echo ""
                        # head -n 2 ${FILENAME} >> logs/summary.log
                        # tail -n 2 ${FILENAME} >> logs/summary.log
                    done
                else
                    FILENAME="logs/${_i}_${PER_Which}_W${W_bit}A32${FOLDIED}.log"
                    echo "Running test case for ${FILENAME}"
                    # python main_myAdaRound.py --scheme_w $_i $PER_CHANNEL_FLAG $FOLDING_FLAG --dstDtypeW $_w | tee ${FILENAME}
                    # echo ""
                    # head -n 2 ${FILENAME} >> logs/summary.log
                    # tail -n 2 ${FILENAME} >> logs/summary.log
                fi
            done
        done
    done
done
