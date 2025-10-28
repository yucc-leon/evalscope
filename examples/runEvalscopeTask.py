from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


# basic_math_names = ['gsm8k','competition_math','cmmlu','ceval']

# Tip: To use the local vLLM SDK instead of a service, set
#   eval_type='vllm_openai' and point `model` to a local path or model ID.
#   Remove `api_url` and `api_key`. Multiprocessing DP can follow examples/test_chat.py
#   by spawning N processes and setting CUDA_VISIBLE_DEVICES per process.
# Example:
# sdk_task_cfg = TaskConfig(
#     model='Qwen/Qwen2.5-0.5B-Instruct',
#     eval_type='vllm_openai',
#     datasets=['cmmlu'],
#     eval_batch_size=1,
#     generation_config={'max_tokens': 512, 'temperature': 0.0, 'top_p': 0.9},
# )
# run_task(task_cfg=sdk_task_cfg)

basic_math_task_cfg = TaskConfig(
    model='zm60b',
    api_url='http://127.0.0.1:8801/v1',
    api_key='EMPTY',
    eval_type=EvalType.VLLMOPENAI,
    datasets=['competition_math','cmmlu','ceval'],
    dataset_args={
        # 'gsm8k': {'few_shot_num': 0},
        'competition_math': {'few_shot_num': 0},
        'cmmlu': {'few_shot_num': 0, 'subset_list': ['college_mathematics', 'high_school_mathematics']},
        'ceval': {'few_shot_num': 0, 'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}
    },
    eval_batch_size=1,
    generation_config={
        'max_tokens': 512,
        'temperature': 0.0,
        'n': 1
    }
)

hard_math_task_cfg = TaskConfig(
    model='zm60b',
    api_url='http://127.0.0.1:8801/v1',
    api_key='EMPTY',
    eval_type=EvalType.VLLMOPENAI,
    datasets=['math_500','aime24','aime25','amc','minerva_math'],
    dataset_args={
        'math_500': {'few_shot_num': 0, },
        'aime24': {'few_shot_num': 0},
        'aime25': {'few_shot_num': 0},
        'amc': {'few_shot_num': 0},
        'minerva_math': {'few_shot_num': 0}
    },
    eval_batch_size=1,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.6,
        'n': 8
    }
)
run_task(task_cfg=basic_math_task_cfg)
run_task(task_cfg=hard_math_task_cfg)
