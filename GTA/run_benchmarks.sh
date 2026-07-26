#!/bin/bash

OUTPUT_FILE="results.json"
if [ "$1" == "mini" ]; then
    NUM_RUNS=1
else
    NUM_RUNS=5
fi

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

# Expand to all 29 experiments
EXPS=($(seq 1 29))

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

    # Shuffle the tuples using shuf to prevent thermal bias
    SHUFFLED_COMBINATIONS=$(printf "%s\n" "${COMBINATIONS[@]}" | shuf)

    # Iterate through the fully randomized list
    for COMB in $SHUFFLED_COMBINATIONS; do
        EXP_ID=$(echo $COMB | cut -d':' -f1)
        LEN=$(echo $COMB | cut -d':' -f2)

        case $EXP_ID in
            1)  NAME="Exp 1: Vanilla MHA (24 Layers)"; ATTN="mha"; LAYERS=24 ;;
            2)  NAME="Exp 2: GQA + RoPE (24 Layers)"; ATTN="gqa"; LAYERS=24 ;;
            3)  NAME="Exp 3: TA + ALiBi (24 Layers)"; ATTN="ta"; LAYERS=24 ;;
            4)  NAME="Exp 4: GQA + RoPE (28 Layers)"; ATTN="gqa"; LAYERS=28 ;;
            5)  NAME="Exp 5: GQA + RoPE (30 Layers)"; ATTN="gqa"; LAYERS=30 ;;
            6)  NAME="Exp 6: TA + ALiBi (28 Layers)"; ATTN="ta"; LAYERS=28 ;;
            7)  NAME="Exp 7: TA + ALiBi (30 Layers)"; ATTN="ta"; LAYERS=30 ;;
            8)  NAME="Exp 8: MHA Iso-KV-Cache (6 Layers)"; ATTN="mha"; LAYERS=6 ;;
            9)  NAME="Exp 9: MHA + RoPE (24 Layers)"; ATTN="mha_rope"; LAYERS=24 ;;
            10) NAME="Exp 10: MHA + RoPE (28 Layers)"; ATTN="mha_rope"; LAYERS=28 ;;
            11) NAME="Exp 11: MHA + RoPE (30 Layers)"; ATTN="mha_rope"; LAYERS=30 ;;
            12) NAME="Exp 12: GTA + ALiBi (24 Layers)"; ATTN="gta"; LAYERS=24 ;;
            13) NAME="Exp 13: GTA + ALiBi (28 Layers)"; ATTN="gta"; LAYERS=28 ;;
            14) NAME="Exp 14: GTA + ALiBi (30 Layers)"; ATTN="gta"; LAYERS=30 ;;
            # --- POSITIONAL ENCODING SWEEPS ---
            15) NAME="Exp 15: MHA + ALiBi (24 Layers)"; ATTN="mha_alibi"; LAYERS=24 ;;
            16) NAME="Exp 16: MHA + ALiBi (28 Layers)"; ATTN="mha_alibi"; LAYERS=28 ;;
            17) NAME="Exp 17: MHA + ALiBi (30 Layers)"; ATTN="mha_alibi"; LAYERS=30 ;;
            18) NAME="Exp 18: GQA + ALiBi (24 Layers)"; ATTN="gqa_alibi"; LAYERS=24 ;;
            19) NAME="Exp 19: GQA + ALiBi (28 Layers)"; ATTN="gqa_alibi"; LAYERS=28 ;;
            20) NAME="Exp 20: GQA + ALiBi (30 Layers)"; ATTN="gqa_alibi"; LAYERS=30 ;;
            21) NAME="Exp 21: TA + PE (24 Layers)"; ATTN="ta_pe"; LAYERS=24 ;;
            22) NAME="Exp 22: TA + PE (28 Layers)"; ATTN="ta_pe"; LAYERS=28 ;;
            23) NAME="Exp 23: TA + PE (30 Layers)"; ATTN="ta_pe"; LAYERS=30 ;;
            24) NAME="Exp 24: GTA + PE (24 Layers)"; ATTN="gta_pe"; LAYERS=24 ;;
            25) NAME="Exp 25: GTA + PE (28 Layers)"; ATTN="gta_pe"; LAYERS=28 ;;
            26) NAME="Exp 26: GTA + PE (30 Layers)"; ATTN="gta_pe"; LAYERS=30 ;;
            27) NAME="Exp 27: GQA + PE (24 Layers)"; ATTN="gqa_pe"; LAYERS=24 ;;
            28) NAME="Exp 28: GQA + PE (28 Layers)"; ATTN="gqa_pe"; LAYERS=28 ;;
            29) NAME="Exp 29: GQA + PE (30 Layers)"; ATTN="gqa_pe"; LAYERS=30 ;;
            *) echo "[!] Unknown EXP_ID '$EXP_ID' — skipping."; continue ;;
        esac
        
        echo "--> Initiating $NAME at Length $LEN"
        python benchmark.py --exp_name "$NAME" --attn_type "$ATTN" --num_layers $LAYERS --prompt_len $LEN
        echo "💤 Sleeping for 3 minutes to stabilize thermals..."
        sleep 180
    done
done