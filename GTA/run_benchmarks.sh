#!/bin/bash

OUTPUT_FILE="results.json"
if [ "$1" == "mini" ]; then
    NUM_RUNS=1
else
    NUM_RUNS=5

# Clear out previous runs
if [ -f "$OUTPUT_FILE" ]; then
    echo "🗑️ Removing old $OUTPUT_FILE"
    rm "$OUTPUT_FILE"
fi

if [ "$1" == "mini" ]; then
    LENGTHS=(512 4096)
else
    LENGTHS=(512 1024 2048 4096)
fi
EXPS=(1 2 3 4 5 6 7)

echo "=========================================="
echo "    STARTING FULLY RANDOMIZED BENCHMARK"
echo "=========================================="

for RUN in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "    STARTING RUN $RUN OF $NUM_RUNS"
    echo "=========================================="

    # Generate all (Experiment, Length) tuples
    COMBINATIONS=()
    for e in "${EXPS[@]}"; do
        for l in "${LENGTHS[@]}"; do
            COMBINATIONS+=("$e:$l")
        done
    done

    # Shuffle the tuples using shuf
    SHUFFLED_COMBINATIONS=$(printf "%s\n" "${COMBINATIONS[@]}" | shuf)

    # Iterate through the fully randomized list
    for COMB in $SHUFFLED_COMBINATIONS; do
        # Extract Exp ID and Length
        EXP_ID=$(echo $COMB | cut -d':' -f1)
        LEN=$(echo $COMB | cut -d':' -f2)

        case $EXP_ID in
            1) 
                NAME="Exp 1: Vanilla MHA (24 Layers)"
                ATTN="mha"
                LAYERS=24 
                ;;
            2) 
                NAME="Exp 2: GQA + RoPE (24 Layers)"
                ATTN="gqa"
                LAYERS=24 
                ;;
            3) 
                NAME="Exp 3: GTA + ALiBi (24 Layers)"
                ATTN="gta"
                LAYERS=24 
                ;;
            4) 
                NAME="Exp 4: GQA + RoPE (28 Layers)"
                ATTN="gqa"
                LAYERS=28 
                ;;
            5) 
                NAME="Exp 5: GQA + RoPE (30 Layers)"
                ATTN="gqa"
                LAYERS=30 
                ;;
            6) 
                NAME="Exp 6: GTA + ALiBi (28 Layers)"
                ATTN="gta"
                LAYERS=28 
                ;;
            7) 
                NAME="Exp 7: GTA + ALiBi (30 Layers)"
                ATTN="gta"
                LAYERS=30 
                ;;
        esac
        
        echo "--> Initiating $NAME at Length $LEN"
        python benchmark.py --exp_name "$NAME" --attn_type "$ATTN" --num_layers $LAYERS --prompt_len $LEN
        echo "💤 Sleeping for 1 minute to stabilize thermals..."
        sleep 60
    done
done

echo "=========================================="
echo "    ALL PROFILES COMPLETE. PLOTTING..."
echo "=========================================="

python plot_results.py