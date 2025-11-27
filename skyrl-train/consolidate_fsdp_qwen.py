import os
import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoConfig
from torch.distributed.tensor import DTensor

# -------------------------------------------------
CKPT_DIR = "/private/schwartz-lab/yarin_shaked7/SkyRL/checkpoints/global_step_85/policy"
HF_META_DIR = os.path.join(CKPT_DIR, "huggingface")
LORA_DIR = os.path.join(CKPT_DIR, "lora_adapter")

# the model you actually trained from
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"   # or local path if offline

OUT_PATH = os.path.join(CKPT_DIR, "consolidated_model.pt")
# -------------------------------------------------


def init_dist():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_base_model():
    # use your saved config so shapes match
    if os.path.exists(os.path.join(HF_META_DIR, "config.json")):
        cfg = AutoConfig.from_pretrained(HF_META_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            config=cfg,
            torch_dtype=torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
        )
    return model


def maybe_attach_lora(model):
    if os.path.isdir(LORA_DIR):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_DIR)
    return model


def main():
    init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # load this rank's shard file
    shard_path = os.path.join(CKPT_DIR, f"model_world_size_{world_size}_rank_{rank}.pt")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(shard_path)

    # IMPORTANT: weights_only=False so pickle can rebuild ShardedTensor
    shard_state = torch.load(shard_path, map_location="cpu", weights_only=False)

    full_state = None
    if rank == 0:
        full_state = {}

    for key, value in shard_state.items():
        if isinstance(value, DTensor):
            value = value.cuda()
            tensor = value.full_tensor()
        else:
            tensor = value

        if rank == 0:
            full_state[key] = tensor.detach().cpu()

    dist.barrier()

    if rank == 0:
        model = load_base_model()
        model = maybe_attach_lora(model)
        missing, unexpected = model.load_state_dict(full_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

        torch.save(model.state_dict(), OUT_PATH)
        print(f"[rank-0] ✅ saved consolidated model to {OUT_PATH}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
