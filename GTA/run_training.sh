#!/bin/bash

# Configuration
SEQ_LEN=512
MICRO_BATCH=2        # Tune this depending on your 8GB VRAM limit
GRAD_ACCUM=16        # Effective batch size = MICRO_BATCH * GRAD_ACCUM (2 * 16 = 32)
OUTPUT_DIR="./training_logs"

mkdir -p "$OUTPUT_DIR"

# Now looping all the way through the 29 architectures
EXPS=($(seq 1 29))

echo "=========================================="
echo "    STARTING FULL CROSS-PE ABLATION SWEEP"
echo "=========================================="

for EXP_ID in "${EXPS[@]}"; do
    case $EXP_ID in
        1)  NAME="Exp_1_Vanilla_MHA_24L"; ATTN="mha"; LAYERS=24 ;;
        2)  NAME="Exp_2_GQA_RoPE_24L"; ATTN="gqa"; LAYERS=24 ;;
        3)  NAME="Exp_3_TA_ALiBi_24L"; ATTN="ta"; LAYERS=24 ;;
        4)  NAME="Exp_4_GQA_RoPE_28L"; ATTN="gqa"; LAYERS=28 ;;
        5)  NAME="Exp_5_GQA_RoPE_30L"; ATTN="gqa"; LAYERS=30 ;;
        6)  NAME="Exp_6_TA_ALiBi_28L"; ATTN="ta"; LAYERS=28 ;;
        7)  NAME="Exp_7_TA_ALiBi_30L"; ATTN="ta"; LAYERS=30 ;;
        8)  NAME="Exp_8_MHA_IsoKV_6L"; ATTN="mha"; LAYERS=6 ;;
        9)  NAME="Exp_9_MHA_RoPE_24L"; ATTN="mha_rope"; LAYERS=24 ;;
        10) NAME="Exp_10_MHA_RoPE_28L"; ATTN="mha_rope"; LAYERS=28 ;;
        11) NAME="Exp_11_MHA_RoPE_30L"; ATTN="mha_rope"; LAYERS=30 ;;
        12) NAME="Exp_12_GTA_ALiBi_24L"; ATTN="gta"; LAYERS=24 ;;
        13) NAME="Exp_13_GTA_ALiBi_28L"; ATTN="gta"; LAYERS=28 ;;
        14) NAME="Exp_14_GTA_ALiBi_30L"; ATTN="gta"; LAYERS=30 ;;
        15) NAME="Exp_15_MHA_ALiBi_24L"; ATTN="mha_alibi"; LAYERS=24 ;;
        16) NAME="Exp_16_MHA_ALiBi_28L"; ATTN="mha_alibi"; LAYERS=28 ;;
        17) NAME="Exp_17_MHA_ALiBi_30L"; ATTN="mha_alibi"; LAYERS=30 ;;
        18) NAME="Exp_18_GQA_ALiBi_24L"; ATTN="gqa_alibi"; LAYERS=24 ;;
        19) NAME="Exp_19_GQA_ALiBi_28L"; ATTN="gqa_alibi"; LAYERS=28 ;;
        20) NAME="Exp_20_GQA_ALiBi_30L"; ATTN="gqa_alibi"; LAYERS=30 ;;
        21) NAME="Exp_21_TA_PE_24L"; ATTN="ta_pe"; LAYERS=24 ;;
        22) NAME="Exp_22_TA_PE_28L"; ATTN="ta_pe"; LAYERS=28 ;;
        23) NAME="Exp_23_TA_PE_30L"; ATTN="ta_pe"; LAYERS=30 ;;
        24) NAME="Exp_24_GTA_PE_24L"; ATTN="gta_pe"; LAYERS=24 ;;
        25) NAME="Exp_25_GTA_PE_28L"; ATTN="gta_pe"; LAYERS=28 ;;
        26) NAME="Exp_26_GTA_PE_30L"; ATTN="gta_pe"; LAYERS=30 ;;
        27) NAME="Exp_27_GQA_PE_24L"; ATTN="gqa_pe"; LAYERS=24 ;;
        28) NAME="Exp_28_GQA_PE_28L"; ATTN="gqa_pe"; LAYERS=28 ;;
        29) NAME="Exp_29_GQA_PE_30L"; ATTN="gqa_pe"; LAYERS=30 ;;
        *) echo "[!] Unknown EXP_ID '$EXP_ID'"; continue ;;
    esac
    
    # Sanitize name to match Python's output format
    SAFE_NAME="${NAME//:/}"
    SAFE_NAME="${SAFE_NAME// /_}"
    
    DONE_FILE="$OUTPUT_DIR/${SAFE_NAME}_DONE.txt"
    LOG_FILE="$OUTPUT_DIR/${SAFE_NAME}_logs.json"

    # Check if the experiment has already been fully completed
    if [ -f "$DONE_FILE" ]; then
        echo "⏭️  [SUCCESS] $NAME is already complete. Skipping."
        continue
    fi

    # Check if artifacts exist from an incomplete run and delete them
    if [ -f "$LOG_FILE" ]; then
        echo "🧹 [CLEANUP] Incomplete run detected for $NAME. Deleting old logs..."
        rm -f "$LOG_FILE"
    fi

    echo "--> Initiating $NAME"
    # max_steps is automatically computed in train.py ensuring exactly 1 Epoch
    python train.py \
        --exp_name "$NAME" \
        --attn_type "$ATTN" \
        --num_layers $LAYERS \
        --seq_len $SEQ_LEN \
        --micro_batch_size $MICRO_BATCH \
        --grad_accum_steps $GRAD_ACCUM \
        --output_dir "$OUTPUT_DIR"
        
    echo "💤 Sleeping for 30 seconds to flush VRAM and stabilize thermals..."
    sleep 30
done

echo "🎉 All 29 training runs completed!"