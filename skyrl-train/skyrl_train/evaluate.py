import torch
from tqdm import tqdm
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from skyrl_train.utils import Timer

from skyrl_train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    prepare_generator_input,
)
from skyrl_train.generators.base import (
    GeneratorOutput,
    GeneratorInterface,
)
from skyrl_train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_gym.envs.lcb.livecodebench import extract_code_from_model

from general_utils import calculate_unique_variable_count, calculate_cyclomatic_complexity, calculate_halstead_volume, calculate_source_lines_of_code

from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (DictConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use

    Returns:
        Dict[str, float]: evaluation metrics
    """

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    sampling_params = cfg.generator.eval_sampling_params
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        validate_generator_output(len(generator_input["prompts"]), generator_output)
        generator_outputs.append(generator_output)
        concat_all_envs.extend(generator_input["env_classes"])
        concat_env_extras.extend(generator_input["env_extras"])
        concat_uids.extend(uids)
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
    vis = tokenizer.decode(generator_output["response_ids"][0])
    logger.info(f"Eval output example: {vis}")

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        concat_generator_outputs, concat_uids, concat_data_sources, cfg.generator.eval_n_samples_per_prompt
    )

    # 3. Calculate overall metrics across all datasets
    overall_avg_score, overall_pass_at_n = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_avg_score,
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_pass_at_n,
        }
    )

    # 4. Compute response length statistics and create wandb table
    responses = concat_generator_outputs["response_ids"]
    rewards = concat_generator_outputs["rewards"]
    prompt_token_ids = concat_generator_outputs["prompt_token_ids"]
    
    num_tokens_arr = np.array([len(response) for response in responses])
    string_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
    
    # Track which responses have code
    extracted_code = [extract_code_from_model(response) for response in string_responses]
    
    # Filter to only responses that have code (for computing averages)
    code_responses_filtered = [code for code in extracted_code if code is not None]
    code_responses = [tokenizer.encode(code, add_special_tokens=False) for code in code_responses_filtered]
    num_code_tokens_arr = np.array([len(code) for code in code_responses])
    unique_variable_count_arr = np.array([calculate_unique_variable_count(code) for code in code_responses_filtered])
    unique_variable_count_arr = unique_variable_count_arr[~np.isnan(unique_variable_count_arr)]
    cyclomatic_complexity_arr = np.array([calculate_cyclomatic_complexity(code) for code in code_responses_filtered])
    cyclomatic_complexity_arr = cyclomatic_complexity_arr[~np.isnan(cyclomatic_complexity_arr)]
    halstead_volume_arr = np.array([calculate_halstead_volume(code) for code in code_responses_filtered])
    halstead_volume_arr = halstead_volume_arr[~np.isnan(halstead_volume_arr)]
    source_lines_of_code_arr = np.array([calculate_source_lines_of_code(code) for code in code_responses_filtered])
    
    # Compute metrics for all responses (for table) - None where code is None
    unique_variable_count_list = []
    cyclomatic_complexity_list = []
    halstead_volume_list = []
    source_lines_of_code_list = []
    
    for code in extracted_code:
        if code is None:
            unique_variable_count_list.append(None)
            cyclomatic_complexity_list.append(None)
            halstead_volume_list.append(None)
            source_lines_of_code_list.append(None)
        else:
            # Calculate metrics, handling NaN values from syntax errors
            uv_count = calculate_unique_variable_count(code)
            unique_variable_count_list.append(None if np.isnan(uv_count) else uv_count)
            
            cc = calculate_cyclomatic_complexity(code)
            cyclomatic_complexity_list.append(None if np.isnan(cc) else cc)
            
            hv = calculate_halstead_volume(code)
            halstead_volume_list.append(None if np.isnan(hv) else hv)
            
            sloc = calculate_source_lines_of_code(code)
            source_lines_of_code_list.append(sloc)
    
    # Support both response-level and token-level rewards
    flat_rewards = []
    for r in rewards:
        if isinstance(r, list):
            flat_rewards.append(float(np.sum(r)))
        else:
            flat_rewards.append(float(r))
    
    string_prompts = tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
    
    # Create wandb table similar to get_rollout_metrics
    summary_table = pd.DataFrame({
        "prompt": string_prompts,
        "code": extracted_code,
        "full_response": string_responses,
        "reward": flat_rewards,
        "unique_variable_count": unique_variable_count_list,
        "cyclomatic_complexity": cyclomatic_complexity_list,
        "source_lines_of_code": source_lines_of_code_list,
        "halstead_volume": halstead_volume_list,
    })
    
    # Calculate average token counts
    avg_num_tokens = np.mean(num_tokens_arr).item()
    avg_code_num_tokens = np.mean(num_code_tokens_arr).item() if len(num_code_tokens_arr) > 0 else 0.0
    avg_unique_variable_count = np.mean(unique_variable_count_arr).item() if len(unique_variable_count_arr) > 0 else 0.0
    avg_cyclomatic_complexity = np.mean(cyclomatic_complexity_arr).item() if len(cyclomatic_complexity_arr) > 0 else 0.0
    avg_halstead_volume = np.mean(halstead_volume_arr).item() if len(halstead_volume_arr) > 0 else 0.0
    avg_source_lines_of_code = np.mean(source_lines_of_code_arr).item() if len(source_lines_of_code_arr) > 0 else 0.0

    # Add response length statistics to eval_metrics
    eval_metrics.update({
        "eval/avg_num_tokens": avg_num_tokens,
        "eval/avg_code_num_tokens": avg_code_num_tokens,
        "eval/avg_unique_variable_count": avg_unique_variable_count,
        "eval/avg_cyclomatic_complexity": avg_cyclomatic_complexity,
        "eval/avg_halstead_volume": avg_halstead_volume,
        "eval/avg_source_lines_of_code": avg_source_lines_of_code,
        "eval/summary_table": summary_table
    })

    import pickle
    import os
    with open(f"{cfg.trainer.export_path}/eval_data.pkl", "wb") as f:
        pickle.dump((generator_outputs, concat_generator_outputs), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()                 # push Python buffer to OS
        os.fsync(f.fileno())      # force OS cache to disk

    # 5. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return eval_metrics
    