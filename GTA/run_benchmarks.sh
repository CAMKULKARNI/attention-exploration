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
EXPS=(1 2 3 4 5 6 7 8 9 10 11)

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
            # -----------------------------------------------------------
            # Exp 8: Iso-KV-Cache MHA baseline.
            # MHA stores num_heads KV heads per layer; GQA stores num_kv_heads.
            # With num_heads=16 and num_kv_heads=4, the ratio is 4:1.
            # So MHA at 6 layers has the same total KV cache as GQA at 24 layers.
            # This isolates the KV-cache memory axis from the layer-count axis.
            # -----------------------------------------------------------
            8)
                NAME="Exp 8: MHA Iso-KV-Cache (6 Layers)"
                ATTN="mha"
                LAYERS=6
                ;;
            # -----------------------------------------------------------
            # Exp 9-11: MHA + RoPE sweep (24 / 28 / 30 layers).
            # Mirrors the GQA and GTA layer sweeps (Exp 2-7).
            # Iso-layer vs Exp 1 (MHA + Learned PE) isolates the cost of
            # swapping positional encoding within full MHA.
            # Iso-layer vs Exp 2 (GQA + RoPE) isolates the cost of head
            # reduction while holding positional encoding constant.
            # -----------------------------------------------------------
            9)
                NAME="Exp 9: MHA + RoPE (24 Layers)"
                ATTN="mha_rope"
                LAYERS=24
                ;;
            10)
                NAME="Exp 10: MHA + RoPE (28 Layers)"
                ATTN="mha_rope"
                LAYERS=28
                ;;
            11)
                NAME="Exp 11: MHA + RoPE (30 Layers)"
                ATTN="mha_rope"
                LAYERS=30
                ;;
            *)
                echo "[!] Unknown EXP_ID '$EXP_ID' — skipping to avoid re-running previous config."
                continue
                ;;
        esac
        
        echo "--> Initiating $NAME at Length $LEN"
        python benchmark.py --exp_name "$NAME" --attn_type "$ATTN" --num_layers $LAYERS --prompt_len $LEN
        echo "💤 Sleeping for 3 minutes to stabilize thermals..."
        sleep 180
    done
done