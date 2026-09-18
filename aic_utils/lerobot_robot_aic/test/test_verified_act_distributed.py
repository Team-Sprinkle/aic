"""CPU checks for conditioning integrity and the ACT distributed update path."""
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


TRAINER = Path(__file__).resolve().parents[3] / "scripts/train_verified_act.py"


def trainer_functions():
    names = {"sample_training_indices", "distributed_sample_indices", "distributed_stop_flags",
             "load_task_inputs", "stable_gaussian_kl"}
    nodes = [node for node in ast.parse(TRAINER.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"np": np, "torch": torch, "dist": dist, "time": time,
             "sys": sys, "Path": Path, "__file__": str(TRAINER)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), scope)
    return scope


def test_global_draw_is_sharded_and_keeps_sampling_streams_aligned():
    sample = trainer_functions()["distributed_sample_indices"]
    groups = [np.arange(500), np.arange(1000, 1500)]
    terminals = [groups[0][-25:], groups[1][-25:]]
    rngs = [np.random.default_rng(16) for _ in range(3)]
    for _ in range(4):
        first = sample(rngs[0], 64, 0, 2, groups, [.3, .7], terminals, .5)
        second = sample(rngs[1], 64, 1, 2, groups, [.3, .7], terminals, .5)
        whole = sample(rngs[2], 128, 0, 1, groups, [.3, .7], terminals, .5)
        np.testing.assert_array_equal(np.concatenate([first, second]), whole)
        assert not np.array_equal(first, second)


def test_conditioning_checks_every_frame_against_metadata(tmp_path):
    sys.path.insert(0, str(TRAINER.parents[1] / "aic_utils/lerobot_robot_aic"))
    from lerobot_robot_aic.task_encoding import encode_task_vector, task_encoding_schema
    load = trainer_functions()["load_task_inputs"]
    tasks = [dict(task_family="sfp_to_nic", target_port_index=1, target_card_index=4, target_card_valid=1),
             dict(task_family="sc_to_sc", target_port_index=0, target_card_index=-1, target_card_valid=0)]
    episodes = np.array([0, 0, 0, 1, 1])
    vectors = np.array([encode_task_vector(**tasks[episode]) for episode in episodes])
    cache = {"task_encoding": task_encoding_schema(),
             "episodes": [{"episode_index": index, "task": task} for index, task in enumerate(tasks)]}
    np.save(tmp_path / "task_vectors.npy", vectors)
    actual, schema = load(tmp_path, cache, episodes)
    np.testing.assert_array_equal(actual, vectors)
    assert schema == task_encoding_schema()
    # A valid one-hot belonging to the wrong task is also rejected.
    vectors[2] = vectors[3]
    np.save(tmp_path / "task_vectors.npy", vectors)
    with pytest.raises(ValueError, match="episode 0"):
        load(tmp_path, cache, episodes)
    cache["task_encoding"]["names"] = list(reversed(cache["task_encoding"]["names"]))
    with pytest.raises(ValueError, match="canonical"):
        load(tmp_path, cache, episodes)


def _distributed_act_worker(rank, rendezvous, output_dir):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import OBS_IMAGES
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    torch.manual_seed(74)
    config = ACTConfig(device="cpu", input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(43,)),
        "observation.images.test": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=2, n_action_steps=1, dim_model=32, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None, dropout=0., vision_backbone="resnet18")
    policy = ACTPolicy(config)
    model = DistributedDataParallel(policy.model, broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    helper = trainer_functions()
    torch.manual_seed(94 + rank)
    initial = policy.model.action_head.weight.detach().clone()
    for step in range(3):
        policy.train()
        data = {"observation.state": torch.randn(2, 43), "action": torch.randn(2, 2, 6),
                "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
                "observation.images.test": torch.randn(2, 3, 32, 32)}
        optimizer.zero_grad(set_to_none=True)
        prediction, (mu, log_variance) = model({**data, OBS_IMAGES: [data["observation.images.test"]]})
        loss = (prediction - data["action"]).abs().mean() + helper["stable_gaussian_kl"](mu, log_variance)
        loss.backward()
        optimizer.step()
        # This reproduces the trainer's asymmetric validation barrier: rank 0
        # uses the original model and never calls a distributed forward alone.
        dist.barrier()
        if rank == 0:
            policy.eval()
            with torch.inference_mode():
                assert policy.predict_action_chunk(data).shape == (2, 2, 6)
        dist.barrier()
    final = policy.model.action_head.weight.detach()
    reference = final.clone()
    dist.broadcast(reference, src=0)
    torch.testing.assert_close(final, reference, rtol=0, atol=0)
    assert not torch.equal(initial, final)
    assert all(not key.startswith("model.module.") for key in policy.state_dict())
    flags = helper["distributed_stop_flags"](rank == 1, float("inf"), time.monotonic(), 100, "cpu", True)
    assert flags == [True, False, False]
    Path(output_dir, f"rank_{rank}.json").write_text(json.dumps({"stop_flags": flags, "updates": 3}))
    dist.destroy_process_group()


def test_two_rank_act_updates_validation_and_one_rank_stop(tmp_path):
    mp.spawn(_distributed_act_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank_{rank}.json").read_text())["updates"] == 3
