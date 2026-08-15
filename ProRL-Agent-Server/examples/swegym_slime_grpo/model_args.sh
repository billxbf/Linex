# shellcheck shell=bash
# Qwen3.5-4B Megatron model args, shared by convert_weights.sh and run.sh.
#
# Mirrors slime/slime/scripts/models/qwen3.5-4B.sh.
# --spec wires in the hybrid GatedDeltaNet + full-attention layer layout.
# tie_word_embeddings=true in HF config → do NOT add --untie-embeddings-and-output-weights.
MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size 2560
    --ffn-hidden-size 9216
    --use-gated-attention
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate
)
