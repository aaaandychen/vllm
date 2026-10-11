# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.spec_decode.mtp_cudagraph import MTPForwardCUDAGraph


def make_metadata(num_reqs=2, query_len=4):
    num_tokens = num_reqs * query_len
    common = CommonAttentionMetadata(
        query_start_loc=torch.arange(0, num_tokens + 1, query_len, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(
            0, num_tokens + 1, query_len, dtype=torch.int32
        ),
        seq_lens=torch.full((num_reqs,), 32, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=query_len,
        max_seq_len=32,
        block_table_tensor=torch.ones((num_reqs, 8), dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
    )
    md = TritonAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=query_len,
        query_start_loc=common.query_start_loc,
        max_seq_len=128,
        seq_lens=common.seq_lens,
        block_table=common.block_table_tensor,
        slot_mapping=common.slot_mapping,
        seq_threshold_3D=16,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(0),
        softmax_segm_max=torch.empty(0),
        softmax_segm_expsum=torch.empty(0),
        causal=True,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    return common, md


@pytest.mark.parametrize("query_len", [1, 4])
def test_replay_stages_new_request_state_in_stable_buffers(query_len):
    graph = MTPForwardCUDAGraph.__new__(MTPForwardCUDAGraph)
    _, target = make_metadata(query_len=query_len)
    common, source = make_metadata(query_len=query_len)
    num_tokens = common.num_actual_tokens
    graph.metadata = {(num_tokens, query_len): {"draft": target}}
    graph.fallback_logged = False
    graph.replay_counts = {}
    graph._forward = lambda *args: args[2]["draft"].seq_lens.clone()
    addresses = [
        getattr(target, field).data_ptr()
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping")
    ]
    for step in range(3):
        source.seq_lens.fill_(40 + step)
        source.block_table.fill_(step + 1)
        source.slot_mapping.add_(1)
        result = graph.replay(common, {"draft": source}, num_tokens)
        torch.testing.assert_close(result, source.seq_lens)
        torch.testing.assert_close(target.block_table, source.block_table)
        torch.testing.assert_close(target.slot_mapping, source.slot_mapping)
    assert addresses == [
        getattr(target, field).data_ptr()
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping")
    ]


@pytest.mark.parametrize(
    "feature",
    ["ragged", "cascade", "table_shape", "uncaptured", "no_cpu_starts", "slots"],
)
def test_unsupported_shape_falls_back_without_staging(feature):
    graph = MTPForwardCUDAGraph.__new__(MTPForwardCUDAGraph)
    _, target = make_metadata()
    common, source = make_metadata()
    graph.metadata = {(8, 4): {"draft": target}}
    graph.fallback_logged = False
    graph.replay_counts = {}
    before = target.seq_lens.clone()
    source.seq_lens.fill_(50)
    if feature == "ragged":
        common.query_start_loc_cpu[1] = 3
    elif feature == "cascade":
        source.use_cascade = True
    elif feature == "table_shape":
        source.block_table = source.block_table[:, :4]
    elif feature == "no_cpu_starts":
        common.query_start_loc_cpu = None
    elif feature == "slots":
        source.slot_mapping = source.slot_mapping[:-1]
    else:
        graph.metadata.clear()
    assert graph.replay(common, {"draft": source}, 8) is None
    torch.testing.assert_close(target.seq_lens, before)


@pytest.mark.parametrize("query_len", [1, 4])
def test_padded_replay_resets_dummy_rows_on_batch_shrink(query_len):
    graph = MTPForwardCUDAGraph.__new__(MTPForwardCUDAGraph)
    _, target = make_metadata(num_reqs=4, query_len=query_len)
    graph.metadata = {(4 * query_len, query_len): {"draft": target}}
    graph.fallback_logged = False
    graph.replay_counts = {}
    graph._forward = lambda n, *args: n
    for num_reqs in (3, 1, 2):
        common, source = make_metadata(num_reqs=num_reqs, query_len=query_len)
        source.slot_mapping.add_(10)
        assert graph.replay(common, {"draft": source}, 4 * query_len) == 4 * query_len
        torch.testing.assert_close(target.seq_lens[:num_reqs], source.seq_lens)
        torch.testing.assert_close(
            target.slot_mapping[: num_reqs * query_len], source.slot_mapping
        )
        assert torch.all(target.seq_lens[num_reqs:] == 0)
        assert torch.all(target.block_table[num_reqs:] == 0)
        assert torch.all(target.slot_mapping[num_reqs * query_len :] == -1)
        assert target.query_start_loc.tolist() == list(
            range(0, 4 * query_len + 1, query_len)
        )
    assert graph.replay_counts[query_len] == [3, 0]


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("query_len", [1, 4])
def test_draft_sizes_and_dispatch_ignore_main_model_rounding(num_reqs, query_len):
    rounded_sizes = [4, 8, 16, 32, 64]
    sizes = MTPForwardCUDAGraph.capture_sizes(rounded_sizes, query_len, 32)
    actual = num_reqs * query_len
    assert actual in sizes
    graph = MTPForwardCUDAGraph.__new__(MTPForwardCUDAGraph)
    _, target = make_metadata(num_reqs=num_reqs, query_len=query_len)
    graph.metadata = {(actual, query_len): {"draft": target}}
    graph.replay_counts = {}
    graph._forward = lambda n, *args: n
    common, source = make_metadata(num_reqs=num_reqs, query_len=query_len)
    padded = next(n for n in rounded_sizes if n >= actual)
    assert graph.replay(common, {"draft": source}, padded) == actual


def test_draft_sizes_respect_capture_limit():
    assert MTPForwardCUDAGraph.capture_sizes([4, 8], 4, 32) == [8, 4]
    assert MTPForwardCUDAGraph.capture_sizes([], 1, 32) == []
    assert max(MTPForwardCUDAGraph.capture_sizes([4, 64], 4, 2)) == 8


@pytest.mark.parametrize(
    "enabled,async_decode", [(True, True), (False, True), (True, False)]
)
def test_accepted_count_copy_only_skipped_for_async_none_cache(enabled, async_decode):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    events = []
    counts_cpu = torch.full((2,), -9, dtype=torch.int32)
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(skip_unused_accepted_tokens_cpu=enabled),
        model_config=SimpleNamespace(is_hybrid=True),
        use_async_spec_decode=async_decode,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        num_accepted_tokens=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        input_batch=SimpleNamespace(num_accepted_tokens_cpu_tensor=counts_cpu),
        num_accepted_tokens_event=SimpleNamespace(record=lambda: events.append(True)),
    )
    tokens = torch.tensor([[3, 4, -1], [5, -1, -1]])
    GPUModelRunner._update_states_after_model_execute(runner, tokens, None)
    torch.testing.assert_close(
        runner.num_accepted_tokens.gpu, torch.tensor([2, 1], dtype=torch.int32)
    )
    if enabled and async_decode:
        assert events == []
        assert counts_cpu.tolist() == [-9, -9]
    else:
        assert events == [True]
        assert counts_cpu.tolist() == [2, 1]
