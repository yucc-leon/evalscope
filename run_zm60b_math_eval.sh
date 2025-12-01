# python examples/run_vllm_sdk_dynamic_samples.py \
#     --model /sharedata/liyuchen/ckpts/zm60b_sft_pp8_ep2_math_sft-test/hf \
#     --model_alias zm60b_sft_test \
#     --task_set quick

python examples/run_vllm_sdk_dynamic_samples.py \
    --model /sharedata/liyuchen/ckpts/zm60b_sft_pp8_ep4_math_sft-math-pack/hf \
    --model_alias math_only \
    --task_set all