#!/bin/bash

# Configuration
MAX_STEPS=10000
SEQ_LEN=512
MICRO_BATCH=2        # Tune this depending on your 8GB VRAM limit
GRAD_ACCUM=16        # Effective batch size = MICRO_BATCH * GRAD_ACCUM (2 * 16 = 32)
OUTPUT_DIR="./training_logs"

mkdir -p "$OUTPUT_DIR"

EXPS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)

echo "=========================================="
echo "    STARTING ARCHITECTURE TRAINING SWEEP"
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
        *) echo "[!] Unknown EXP_ID '$EXP_ID'"; continue ;;
    esac
    
    echo "--> Initiating $NAME"
    python train.py \
        --exp_name "$NAME" \
        --attn_type "$ATTN" \
        --num_layers $LAYERS \
        --max_steps $MAX_STEPS \
        --seq_len $SEQ_LEN \
        --micro_batch_size $MICRO_BATCH \
        --grad_accum_steps $GRAD_ACCUM \
        --output_dir "$OUTPUT_DIR"
        
    echo "💤 Sleeping for 60 seconds to flush VRAM and stabilize thermals..."
    sleep 60
done

echo "🎉 All training runs completed!"
