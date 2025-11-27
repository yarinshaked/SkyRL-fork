from tqdm import tqdm
from skyrl_gym.envs.lcb.livecodebench import compute_score
import pickle


with open("spartan_baseline_evaluation.pkl", "rb") as f:
    eval_df = pickle.load(f)


for idx, (reponse, tests) in tqdm(enumerate(zip(eval_df["response"], eval_df["reward_spec"])), desc="Evaluating responses"):
    tests = eval(tests["ground_truth"])
    parsed_code, reward = compute_score(reponse, tests)
    eval_df.loc[idx, "parsed_code"] = parsed_code
    eval_df.loc[idx, "reward"] = reward
    idx += 1


print(eval_df.reward.value_counts())