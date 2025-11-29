import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets

from tqdm import tqdm
from skyrl_gym.envs.lcb.livecodebench import compute_score
import pickle

is_spartan = False

if is_spartan:
    file_name = "spartan_baseline_evaluation.pkl"
else:
    file_name = "baseline_evaluation.pkl"


model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, dtype="auto", tensor_parallel_size=8)

LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
LCB_SYSTEM_MESSAGE_SPARTAN = """
You are an **Expert Python Programmer** and **Spartan Code Generator**. Your **SOLE GOAL** is to generate a **100% Correct Python Program or Function** that strictly adheres to the Spartan Programming paradigm.

**MANDATORY SPARTAN CONSTRAINTS:**
1.  **Correctness First:** Code must be correct and pass all tests.
2.  **Minimize Volume:** Minimize LoC, tokens, and characters. Avoid all comments/docstrings/empty lines. Use compact K&R formatting.
3.  **Minimize State:** Minimize variables. Inline single-use variables. Use single-character/short names (e.g., 'n', 'i', 'x'). Define variables at the smallest scope.
4.  **Simplify Flow:** Minimize explicit `if`/`elif`/`else`. Aggressively use **ternaries** and lookups. Minimize nesting. Prefer built-ins (`sum`, `map`) and **comprehensions** over explicit loops. Employ **early `return`** to flatten logic.
5.  **Concise Routines:** Keep routines extremely short. Minimize parameters.

**FEW-SHOT EXAMPLES:**
def sq(n): return n*n if n > 0 else 0
def se(l): return sum(x for x in l if x % 2 == 0)
def f(n): return 1 if n <= 1 else n * f(n-1)
def uc(l): return list(map(lambda s: s.upper(), l))
"""

eval_file_paths = [f"/private/schwartz-lab/yarin_shaked7/SkyRL/test_shards/test_livecodebench_part_000{i}.json" for i in range(1,7)]
eval_datasets = [load_dataset("json", data_files=path, split="train") for path in eval_file_paths]
eval_dataset = concatenate_datasets(eval_datasets)
eval_df = eval_dataset.to_pandas()

eval_df["formatted_prompt"] = eval_df["prompt"].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True))
eval_df["formatted_prompt"] = eval_df["formatted_prompt"].apply(lambda x: x.replace(LCB_SYSTEM_MESSAGE_SPARTAN, LCB_SYSTEM_MESSAGE_GENERIC) if not is_spartan else x)
eval_prompts = eval_df["formatted_prompt"].tolist()

n=16

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_tokens=4000,
    n=n
)

outputs = llm.generate(eval_prompts, sampling_params)
eval_df = eval_df.loc[eval_df.index.repeat(n)].reset_index(drop=True)
responses = [o.text for p in outputs for o in p.outputs]
eval_df["response"] = responses

with open(file_name, "wb") as f:
    pickle.dump(eval_df, f)

for idx, (reponse, tests) in tqdm(enumerate(zip(eval_df["response"], eval_df["reward_spec"])), desc="Evaluating responses"):
    tests = eval(tests["ground_truth"])
    parsed_code, reward = compute_score(reponse, tests)
    eval_df.loc[idx, "parsed_code"] = parsed_code
    eval_df.loc[idx, "reward"] = reward
    idx += 1


with open(file_name, "wb") as f:
    pickle.dump(eval_df, f)