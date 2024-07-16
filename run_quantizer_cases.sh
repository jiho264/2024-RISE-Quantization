SCHEMES=( "AbsMaxQuantizer" "MinMaxQuantizer" "NormQuantizer" "OrgNormQuantizerCode" )
DSTDTYPEW=( "INT4" "INT8" )
PERCHANNEL=( "False" "True" )
L_PNORM=( "2" "2.4")

for _i in "${SCHEMES[@]}"; do
    for _j in "${PERCHANNEL[@]}"; do
        if [ "$_j" == "False" ]; then
            PER_Which="Layer"
            PER_CHANNEL_FLAG=""
        else
            PER_Which="CH"
            PER_CHANNEL_FLAG="--per_channel"
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
                    FILENAME="logs/${_i}_${PER_Which}_W${W_bit}A32_p${p_norm}.log"
                    echo "Running test case for ${FILENAME}"
                    python main_myAdaRound.py --scheme $_i $PER_CHANNEL_FLAG --dstDtypeW $_w --p ${_p} | tee ${FILENAME}
                    echo ""
                done
            else
                FILENAME="logs/${_i}_${PER_Which}_W${W_bit}A32.log"
                echo "Running test case for ${FILENAME}"
                python main_myAdaRound.py --scheme $_i $PER_CHANNEL_FLAG --dstDtypeW $_w | tee ${FILENAME}
                echo ""
            fi
        done
    done
done
