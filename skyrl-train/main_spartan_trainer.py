import ray
import hydra
import torch
from typing import List
from omegaconf import DictConfig
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg

from skyrl_train.generators.base import GeneratorOutput
import re
import numpy as np
from general_utils import calculate_halstead_volume
    

class SpartanTrainer(RayPPOTrainer):
    def extract_code_from_model(self, model_response: str):
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
        if not code_blocks:
            return None
        return code_blocks[-1].strip()

    def calculate_spartan_metrics(self, code: str):
        return calculate_halstead_volume(code)
    
    @torch.no_grad()
    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        spartan_reward_coef = self.cfg.trainer.algorithm.spartan_reward_coef
        format_reward_coef = self.cfg.trainer.algorithm.format_reward_coef
        accuracy_reward_coef = self.cfg.trainer.algorithm.accuracy_reward_coef
        rollout_size = self.cfg.generator.n_samples_per_prompt

        # --- decode responses ---
        response_ids = generator_output["response_ids"]
        responses = [self.tokenizer.decode(rids, skip_special_tokens=True) for rids in response_ids]

        # extract code if fenced
        codes = []
        has_code_block_list = []
        for resp in responses:
            code = self.extract_code_from_model(resp)
            codes.append(code if code is not None else resp)
            has_code_block_list.append(code is not None)

        spartan_metrics = [self.calculate_spartan_metrics(code) for code in codes]
        rewards = generator_output["rewards"]

        # get correctness mask (accuracy)
        is_correct_list = []
        for r in rewards:
            # underlying reward==0 ==> not correct, else correct
            if isinstance(r, list):
                is_correct_list.append(bool(r[-1]))
            else:
                is_correct_list.append(bool(r))

        # --- Prepare individual reward lists ---
        accuracy_rewards = []    # configurable accuracy for correct, 0.0 for incorrect
        format_rewards = []      # format delta for each sample

        num_samples = len(spartan_metrics)
        new_rewards = list(rewards)  # Mutable copy for "rewards"

        # Iterate over the batch in chunks of rollout_size
        for start_idx in range(0, num_samples, rollout_size):
            end_idx = min(start_idx + rollout_size, num_samples)
            rollout_spartan_metrics = spartan_metrics[start_idx:end_idx]
            rollout_is_correct = is_correct_list[start_idx:end_idx]
            rollout_has_code_block = has_code_block_list[start_idx:end_idx]

            # --- choose a baseline length and compute std for the current rollout ---
            correct_lengths = [
                M for M, ok in zip(rollout_spartan_metrics, rollout_is_correct) if ok
            ]
            if correct_lengths:
                lengths_for_stats = correct_lengths
            else:
                lengths_for_stats = rollout_spartan_metrics if rollout_spartan_metrics else [1.0]
            mean_len = float(np.mean(lengths_for_stats))
            std_len = float(np.std(lengths_for_stats))
            # Prevent division by zero
            std_len = max(std_len, 1.0)

            for j in range(len(rollout_spartan_metrics)):
                i = start_idx + j  # Global index

                has_code = rollout_has_code_block[j]
                base_ok = rollout_is_correct[j]

                if not has_code:
                    accuracy_reward = -accuracy_reward_coef
                    format_reward = 0.0
                    spartan_reward = 0.0
                elif has_code and not base_ok:
                    accuracy_reward = -accuracy_reward_coef
                    format_reward = format_reward_coef
                    spartan_reward = 0.0
                elif has_code and base_ok:
                    accuracy_reward = accuracy_reward_coef
                    rel = (mean_len - spartan_metrics[i]) / std_len
                    spartan_reward = max(-spartan_reward_coef, spartan_reward_coef * rel)
                    format_reward = format_reward_coef
                
                new_reward = accuracy_reward + format_reward + spartan_reward

                # Log component rewards
                format_rewards.append(float(format_reward))
                accuracy_rewards.append(float(accuracy_reward))

                if isinstance(new_rewards[i], list):
                    new_rewards[i][-1] = new_reward
                else:
                    new_rewards[i] = new_reward

        # Update the generator output with all reward components (except rewards-length)
        generator_output["rewards"] = new_rewards
        generator_output["rewards-accuracy"] = accuracy_rewards
        generator_output["rewards-format"] = format_rewards
        return super().postprocess_generator_output(generator_output, uids)

class SpartanExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return SpartanTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = SpartanExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()