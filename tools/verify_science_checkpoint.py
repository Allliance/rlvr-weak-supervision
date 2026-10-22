"""CPU validation of every shard and restored counter for this one-GPU protocol."""
import argparse
import json
import math
import os
from pathlib import Path


def validate(checkpoint, model_path):
    assert all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    step = int(checkpoint.name.rsplit("_", 1)[1])
    assert int((checkpoint.parent / "latest_checkpointed_iteration.txt").read_text()) >= step

    def load(relative):
        return torch.load(checkpoint / relative, map_location="cpu", weights_only=False, mmap=True)

    def finite(tensor):
        tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        assert tensor.device.type == "cpu"
        flat = tensor.reshape(-1)
        for start in range(0, flat.numel(), 2**22):
            assert bool(torch.isfinite(flat[start:start + 2**22]).all()), "Non-finite checkpoint tensor"
        return tensor

    shapes = {}
    for path in model_path.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as archive:
            shapes.update({k: tuple(archive.get_slice(k).get_shape()) for k in archive.keys()})
    assert shapes
    model = load("actor/model_world_size_1_rank_0.pt")
    tied = json.loads((model_path / "config.json").read_text()).get("tie_word_embeddings", False)
    assert set(shapes) <= set(model)
    assert not (set(model) - set(shapes) - ({"lm_head.weight"} if tied else set()))
    for name, tensor in model.items():
        assert tuple(finite(tensor).shape) == shapes.get(name, shapes["model.embed_tokens.weight"])
    if tied and "lm_head.weight" in model:
        assert torch.equal(finite(model["lm_head.weight"]), finite(model["model.embed_tokens.weight"]))
    extra = load("actor/extra_state_world_size_1_rank_0.pt")
    assert extra["lr_scheduler"]["last_epoch"] == step
    assert {"cpu", "cuda", "numpy", "random"} <= set(extra["rng"])
    data = load("data.pt")
    assert isinstance(data, dict) and data
    optimizer = load("actor/optim_world_size_1_rank_0.pt")
    order = [k for group in optimizer["param_groups"] for k in group["params"]]
    assert len(order) == len(set(order)) and set(optimizer["state"]) <= set(order)
    for group in optimizer["param_groups"]:
        assert math.isclose(group["lr"], 1e-6, rel_tol=1e-6)
    missing = set(order) - set(optimizer["state"])
    if missing:
        assert tied and len(order) == len(model)
        assert {list(model)[order.index(k)] for k in missing} == {"lm_head.weight"}
    for value in optimizer["state"].values():
        assert {"step", "exp_avg", "exp_avg_sq"} <= set(value)
        assert float(finite(value["step"])) == step
        assert finite(value["exp_avg"]).shape == finite(value["exp_avg_sq"]).shape
    return dict(status="passed", checkpoint=str(checkpoint), step=step,
                model_tensors=len(model), initialized_adam_states=len(optimizer["state"]),
                scheduler_step=extra["lr_scheduler"]["last_epoch"], rng_keys=sorted(extra["rng"]),
                dataloader_keys=sorted(data), all_tensors_finite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.checkpoint, args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
