import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

llm = LLM(model=model_name, dtype="auto", tensor_parallel_size=4, trust_remote_code=True)
sampling_params = SamplingParams(
    repetition_penalty=1.0,
    temperature=0,
    top_p=1,
    top_k=-1,
    min_p=0.0,
    max_tokens=4000,
)

prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
]

outputs = llm.generate(prompts, sampling_params)