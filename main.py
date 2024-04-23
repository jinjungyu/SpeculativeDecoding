
import torch
import argparse
import contexttimer
from transformers import AutoTokenizer

from sampling import autoregressive_sampling, autoregressive_sampling_FT, speculative_sampling, speculative_sampling_v2
import json
from  tqdm import tqdm
import gzip
import numpy as np
from torch.utils.data import Dataset
import random

from utils.globals import Decoder
from utils.misc import seed_all
from utils.modelutils import get_hfmodel, model_multigpu

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target_model_name', type=str, default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--approx_model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--target_model_load', type=str, default=None)
    parser.add_argument('--approx_model_load', type=str, default=None)
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--ft', action='store_true', default=False, help='using fastertransformer')
    
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--input', type=str, default="Suggest at least five related search terms to \"Mạng neural nhân tạo\".")
    args = parser.parse_args()
    return args


def benchmark(fn, info, *args, **kwargs):
    questions_dict = {}
    answers = []
    with open('./vicuna_questions.jsonl') as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            questions_dict[q["question_id"]] = q["text"]
            answers.append(q)
    
    prompt = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "### Human: {user_question}"
        "### Assistant: "
    )
    # with gzip.open("./data/enwik8.gz") as file:
    #     data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    #     _, np_valid = np.split(data, [int(90e6)])
    #     data_val = torch.from_numpy(np_valid)

    # class TextSamplerDataset(Dataset):
    #     def __init__(self, data, seq_len):
    #         super().__init__()
    #         self.data = data
    #         self.seq_len = seq_len

    #     def __getitem__(self, index):
    #         rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
    #         full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
    #         return full_seq.to('cuda')

    #     def __len__(self):
    #         return self.data.size(0) // self.seq_len

    # SEQ_LEN = 128
    # dataset = TextSamplerDataset(data_val, SEQ_LEN)
    verbose = kwargs.pop('verbose')
        
    test_sample_num = 5
    with contexttimer.Timer() as t:
        total_tokens = 0
        total_accept = 0
        total_reject = 0
        total_gen_minus_all_accept = 0
        for i in tqdm(range(test_sample_num), desc=f"{info} benchmarking"):
            qid = random.randint(0,len(questions_dict))
            # input_ids = random.choice(dataset).unsqueeze(0)
            # input_text = Decoder().decode(input_ids)
            in_prompt = prompt.format(user_question=questions_dict[qid])
            input_ids = Decoder().encode(in_prompt)
            if info == 'SP':
                output_ids, gen_minus_all_accept, accept, reject = fn(input_ids, *args, **kwargs)
                total_accept += accept
                total_reject += reject
                total_gen_minus_all_accept += gen_minus_all_accept
            else:
                output_ids  = fn(input_ids, *args, **kwargs)
            total_tokens += (output_ids.shape[-1] - input_ids.shape[-1])

            generated_text = Decoder().decode(output_ids[...,-(output_ids.shape[-1] - input_ids.shape[-1]):])
            
            if verbose:
                print("== input text ==")
                print(questions_dict[qid])
                print()
                print("== generated_text ==\n")
                print(generated_text)
                print("====================\n")
                
    if info == 'SP':
        print(f"Accepatance rate : {total_accept / total_gen_minus_all_accept}")
    # print(f"\n [benchmark] {info} tokens/sec: {total_tokens / t.elapsed}, {t.elapsed} sec generates {total_tokens} tokens")
    return total_tokens, t.elapsed

def generate(approx_model_name, target_model_name, 
             approx_model_load = None, target_model_load = None,
             num_tokens=100, gamma = 4,
             random_seed = None, verbose = False, ft = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)

    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    
    # TODO : FasterTransformer LLaMA
        
    # approx model
    approx_model = get_hfmodel(approx_model_name)
    
    # target model
    target_model = get_hfmodel(target_model_name)
    
    # 1. fp16 fp16, 2. fp16 W4
    if target_model_load is None:
        target_model_gpus = [torch.device(f'cuda:{gpu_id}') for gpu_id in range(torch.cuda.device_count())] # 앞 4개
        approx_model_gpu = torch.device(f'cuda:{len(target_model_gpus)-1}') # 뒤 1개
            
        if ft:
            target_model.device = target_model_gpus[0]
            approx_model.device = approx_model_gpu
        
        model_multigpu(target_model, target_model_gpus, model_name=target_model_name)
        approx_model.to(approx_model_gpu)
    # 3. W4 W4
    else:
        gpu = torch.device('cuda:0')
        target_model.to(gpu)
        approx_model.to(gpu)

        if ft:
            target_model.device = gpu
            approx_model.device = gpu
            
    print("finish loading models")
    # import code; code.interact('After Loading', local=dict(globals(), **locals()))
    as_fn = autoregressive_sampling_FT if ft else autoregressive_sampling
    top_k = 10
    top_p = 0.95
    temperature = 0.8
    
    seed_all(123)
    num_gen_tokens_target, elapsed_time_target = benchmark(as_fn, "AS_target", target_model, num_tokens, top_k = top_k, top_p=top_p, temperature = temperature, **{'verbose': args.verbose})
    throuput_target = num_gen_tokens_target / elapsed_time_target
    torch.cuda.empty_cache()

    # seed_all(123)
    # num_gen_tokens_approx, elapsed_time_approx = benchmark(as_fn, "AS_approx", approx_model, num_tokens, top_k = top_k, top_p=top_p)
    # throuput_approx = num_gen_tokens_approx / elapsed_time_approx
    # torch.cuda.empty_cache()
    
    seed_all(123)
    num_gen_tokens_sp, elapsed_time_sp = benchmark(speculative_sampling, "SP", approx_model, target_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, temperature = temperature, ft=ft, **{'verbose': args.verbose})
    throuput_sp = num_gen_tokens_sp / elapsed_time_sp
    
    print(f"Target model Throuput : {throuput_target} Token/s")
    # print(f"Approx model Throuput : {throuput_approx} Token/s")
    print(f"SP Throuput : {throuput_sp} Token/s.")
    print(f"SP Speed up : {throuput_sp / throuput_target}.")
if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.approx_model_name, args.target_model_name, 
             args.approx_model_load, args.target_model_load, 
             num_tokens=args.max_tokens, gamma=args.gamma, verbose=args.verbose, ft=args.ft)
