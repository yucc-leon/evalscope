from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.sampling_params import SamplingParams
import os, glob
from tqdm import tqdm
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_HOME"] = "/usr/local/cuda"
# os.environ["VLLM_PP_LAYER_PARTITION"] = "4,4,4,4,4,4,4,2"
# os.environ["VLLM_PP_LAYER_PARTITION"] = "8,8,8,6"
import torch

import os, json, multiprocessing as mp
from vllm import LLM, SamplingParams

MODEL_PATH   = "/sharedata/zimoliu/ckpts/jamba_60b_aws_oh_pp8_ep4_efa_512k_sft_v1_16node_ckpt75000/hf"
SAMPLING_KWARGS = dict(max_tokens=2048,
    temperature=0.75,
    top_p=0.9,
    # top_k=1
    )

def work(gpu_id: int, q: mp.Queue):
    """子进程入口，独占一张卡"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = LLM(
        model=MODEL_PATH,
        # tensor_parallel_size=1,
        # gpu_memory_utilization=0.85,
        # enforce_eager=True,
        # trust_remote_code=True
    )
    sampling_params = SamplingParams(**SAMPLING_KWARGS)

    def chat(user_input="你好，你是谁？"):

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
        # outputs = llm.generate(prompts=user_input, sampling_params=sampling_params)
        outputs = llm.chat([conversation], sampling_params, use_tqdm=True)
        assistant_reply = outputs[0].outputs[0].text.strip()
        print(f"\n{gpu_id} [Input]: {user_input}\n[Output]: {assistant_reply}")
        return assistant_reply
    chat()


    test_data_dir = "/workspace/zimoliu/code/UltraEval/datasets/cmmlu/data/"
    result_dir = "/workspace/zimoliu/code/vllm/examples/offline_inference/notebook/result/cmmlu/"

    input_files = glob.glob(os.path.join(test_data_dir, "**/*.json"), recursive=True) + glob.glob(os.path.join(test_data_dir, "**/*.jsonl"), recursive=True)
    print("### [rank {}] input files: {}\n".format(gpu_id, input_files))

    all_total_q_cnt, all_total_r_cnt = 0, 0
    all_total_ratio_lst = []
    dp_size = 8
    for file_id, input_file_name in enumerate(input_files):
        if file_id % dp_size != gpu_id:
            continue
        #     data = data + data[-(dp_size - (len(data) % dp_size)):]
        # assert len(data) % dp_size == 0, "data len not match dp size."
        data = []
        print("### rank: {} processing file: {}\n".format(gpu_id, input_file_name))
        with open(input_file_name, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                js = json.loads(l)
                choices = []
                answer_id = None
                for i, (kw, val) in enumerate(js["target_scores"].items()):
                    if val > 0:
                        answer_id = i
                    choices.append(kw)
                if answer_id is None:
                    raise ValueError("Invalid data `{}`".format(js))
                # question = js['question']
                
                choice_style = "({})"

                sep_style = " "

                idx_style = "ABCD"

                prompt_base = "题目：\n{question}\n\n选项：\n{options}\n\n请解答上面的选择题，直接返回正确答案的选项字母。\n答案："

                options = []
                for i, choice in enumerate(choices):
                    options.append(choice_style.format(idx_style[i]) + sep_style + choice)

                prompt_base = prompt_base.format(question=js["question"], options="\n".join(options))


                true_answer = idx_style[answer_id]

                data.append({"input_prompts": prompt_base, "true_answer": true_answer, "true_answer_idx": answer_id})

        print("### rank: {} process file: {} succeed, len: {}\n".format(gpu_id, input_file_name, len(data)))
        
        q_cnt, r_cnt = 0, 0
        # for id, js in enumerate(data):
        for id, js in tqdm(enumerate(data), total=len(data),
                   desc=f"rank{gpu_id} {input_file_name.rsplit('/',1)[-1]}"):
            
            reply = chat(user_input=js["input_prompts"])
            q_cnt = q_cnt + 1
            if js["true_answer"] in reply:
                r_cnt = r_cnt + 1
                js["reply"] = reply
                js["corret"] = 1
            else:
                js["corret"] = 0
            
            print("### rank:{}, file : {}, data js: {}, q num: {}, right num: {}, ratio: {}\n".format(gpu_id, input_file_name, js, q_cnt, r_cnt, float(r_cnt)/float(q_cnt)))

            tqdm.write(f"rank{gpu_id}  {q_cnt}/{len(data)}  "
               f"right={r_cnt}  acc={r_cnt/q_cnt:.1%}")
            
        all_total_q_cnt = all_total_q_cnt + q_cnt
        all_total_r_cnt = all_total_r_cnt + r_cnt
        total_ratio = float(r_cnt) / float(q_cnt)
        all_total_ratio_lst.append(total_ratio)
        
        print("### rank:{}, writing data and results for file: {}, all_total_q_cnt: {}, all_total_r_cnt: {}, all_total_ratio for now: {}. \n".format(gpu_id, input_file_name, all_total_q_cnt, all_total_r_cnt, (float(all_total_r_cnt) / float(all_total_q_cnt))))

        with open(result_dir + input_file_name.rsplit("/",1)[-1], 'a') as f:
            for item in data:          # data 是 list[dict]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            # f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # return (all_total_q_cnt, all_total_r_cnt)
    q.put((all_total_q_cnt, all_total_r_cnt))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    ps = [mp.Process(target=work, args=(i, q)) for i in range(2)]

    for p in ps: p.start()
    for p in ps: p.join()

    # 收集 8 份结果
    total_q = total_r = 0
    for _ in range(8):
        q_cnt, r_cnt = q.get()
        total_q += q_cnt
        total_r += r_cnt

    print(">>> 全部推理完成 <<<")
    print(f"总计题目：{total_q}，正确：{total_r}，整体正确率：{total_r/total_q:.4%}")
    