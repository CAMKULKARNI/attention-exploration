#!/bin/bash
set -euo pipefail

SEQ_LEN=512
MICRO_BATCH=2
EVAL_BATCH=16
GRAD_ACCUM=16
OUTPUT_DIR="./ablation_logs"
TRAIN_FILE="wikitext103_train.bin"
VAL_FILE="wikitext103_val.bin"

mkdir -p "$OUTPUT_DIR"

# Explicit 28 Configurations Preserving Full Granular Ablation Matrix:
# Format: "LAYERS USE_Q USE_K USE_V ACT_MODE"
# ACT_MODE: "none" (baseline), or target projection keys to receive non-linearity ("all", "qk", "qv", "kv", "q", "k", "v")
CONFIGS=(
    "24 true  true  true  none"
    "24 true  true  true  all"
    "24 false false false none"
    "30 false false false none"
    "24 true  true  false none"
    "24 true  true  false qk"
    "26 true  true  false none"
    "26 true  true  false qk"
    "24 true  false true  none"
    "24 true  false true  qv"
    "26 true  false true  none"
    "26 true  false true  qv"
    "24 false true  true  none"
    "24 false true  true  kv"
    "26 false true  true  none"
    "26 false true  true  kv"
    "24 true  false false none"
    "24 true  false false q"
    "28 true  false false none"
    "28 true  false false q"
    "24 false true  false none"
    "24 false true  false k"
    "28 false true  false none"
    "28 false true  false k"
    "24 false false true  none"
    "24 false false true  v"
    "28 false false true  none"
    "28 false false true  v"
)

echo "================================================="
echo " STARTING VERIFIED PROJECTION ABLATIONS          "
echo "================================================="

for CFG in "${CONFIGS[@]}"; do
    read -r LAYERS USE_Q USE_K USE_V ACT_MODE <<< "$CFG"

    # Sweep non-linearities only for configurations that target activations
    if [ "$ACT_MODE" = "none" ]; then
        NONLIN_LIST=("none")
    else
        NONLIN_LIST=("gelu" "bottleneck")
    fi

    for NONLIN_TYPE in "${NONLIN_LIST[@]}"; do
        Q_ACT="none"
        K_ACT="none"
        V_ACT="none"

        if [ "$NONLIN_TYPE" != "none" ]; then
            case "$ACT_MODE" in
                "all") Q_ACT="$NONLIN_TYPE"; K_ACT="$NONLIN_TYPE"; V_ACT="$NONLIN_TYPE" ;;
                "qk")  Q_ACT="$NONLIN_TYPE"; K_ACT="$NONLIN_TYPE" ;;
                "qv")  Q_ACT="$NONLIN_TYPE"; V_ACT="$NONLIN_TYPE" ;;
                "kv")  K_ACT="$NONLIN_TYPE"; V_ACT="$NONLIN_TYPE" ;;
                "q")   Q_ACT="$NONLIN_TYPE" ;;
                "k")   K_ACT="$NONLIN_TYPE" ;;
                "v")   V_ACT="$NONLIN_TYPE" ;;
            esac
        fi

        for IS_GQA in false true; do
            # Skip invalid GQA topologies where K or V cannot form grouped projections
            if [ "$IS_GQA" = "true" ] && { [ "$USE_K" = "false" ] || [ "$USE_V" = "false" ]; }; then
                continue
            fi

            MODE_NAME="MHA"
            [ "$IS_GQA" = "true" ] && MODE_NAME="GQA"

            NAME="Exp_L${LAYERS}_Q${USE_Q}_K${USE_K}_V${USE_V}_ActMode${ACT_MODE}_${MODE_NAME}_${NONLIN_TYPE}"
            DONE_FILE="$OUTPUT_DIR/${NAME}_DONE.txt"

            if [ -f "$DONE_FILE" ]; then
                echo "⏭️  [SKIP] $NAME is already complete."
                continue
            fi

            echo "--> Initiating $NAME (Layers: $LAYERS, Q:$USE_Q, K:$USE_K, V:$USE_V, ActMode: $ACT_MODE, Nonlin: $NONLIN_TYPE)"
            
            python train.py \
                --exp_name "$NAME" \
                --num_layers "$LAYERS" \
                --is_gqa "$IS_GQA" \
                --use_q_proj "$USE_Q" \
                --use_k_proj "$USE_K" \
                --use_v_proj "$USE_V" \
                --q_act "$Q_ACT" \
                --k_act "$K_ACT" \
                --v_act "$V_ACT" \
                --seq_len "$SEQ_LEN" \
                --micro_batch_size "$MICRO_BATCH" \
                --eval_batch_size "$EVAL_BATCH" \
                --grad_accum_steps "$GRAD_ACCUM" \
                --train_data_file "$TRAIN_FILE" \
                --val_data_file "$VAL_FILE" \
                --output_dir "$OUTPUT_DIR"

            sleep 2
        done
    done
done

echo "🎉 All ablations executed with absolute benchmarking and algorithmic integrity."