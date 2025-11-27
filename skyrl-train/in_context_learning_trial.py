import os
os.environ["NCCL_P2P_DISABLE"] = "1"
import pickle
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from skyrl_gym.envs.lcb.livecodebench import compute_score

# Generate responses function
def generate_responses(df, llm, sampling_params, n_rows=1):
    mini_df = df.iloc[:n_rows]
    inputs = mini_df["formatted_prompt"].to_list()
    outputs = llm.generate(inputs, sampling_params)
    full_responses = [[sequence.text for sequence in output.outputs] for output in outputs]
    
    codes, rewards = [], []
    for tests, responses in zip(mini_df["reward_spec"], full_responses):
        for text in responses:
            parsed_code, reward = compute_score(text, eval(tests['ground_truth']))
            codes.append(parsed_code)
            rewards.append(reward)

    return mini_df, full_responses, codes, rewards


if __name__ == '__main__':
    # Setup CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    NUM_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES").count(",") + 1

    # Initialize LLM
    llm = LLM(
        model="Qwen/Qwen3-4B",
        tensor_parallel_size=NUM_GPUS
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataframes
    vanilla_df = pd.read_pickle("united_train_dataset.pkl")
    spartan_df = pd.read_pickle("united_spartan_train_dataset.pkl")

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=4000,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        n=8
    )

    n_rows = 100

    # Generate responses for vanilla dataset
    print("Generating responses for vanilla dataset...")
    mini_vanilla_df, full_vanilla_responses, vanilla_codes, vanilla_rewards = generate_responses(
        vanilla_df, llm, sampling_params, n_rows=n_rows
    )

    # Save vanilla outputs as tuple
    vanilla_outputs = (mini_vanilla_df, full_vanilla_responses, vanilla_codes, vanilla_rewards)
    with open("vanilla_generate_responses_outputs.pkl", "wb") as f:
        pickle.dump(vanilla_outputs, f)
    print("Saved vanilla outputs to vanilla_generate_responses_outputs.pkl")

    # Generate responses for spartan dataset
    print("Generating responses for spartan dataset...")
    mini_spartan_df, full_spartan_responses, spartan_codes, spartan_rewards = generate_responses(
        spartan_df, llm, sampling_params, n_rows=n_rows
    )

    # Save spartan outputs as tuple
    spartan_outputs = (mini_spartan_df, full_spartan_responses, spartan_codes, spartan_rewards)
    with open("spartan_generate_responses_outputs.pkl", "wb") as f:
        pickle.dump(spartan_outputs, f)
    print("Saved spartan outputs to spartan_generate_responses_outputs.pkl")

    print("All outputs saved successfully!")

