# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

logger = init_logger(__name__)


class MTPForwardCUDAGraph:
    """Full forward graphs with graph-owned Triton attention metadata."""

    def __init__(self, proposer: Any):
        self.proposer = proposer
        self.wrapper = CUDAGraphWrapper(
            proposer.model, proposer.vllm_config, CUDAGraphMode.FULL
        )
        self.metadata: dict[tuple[int, int], dict[str, TritonAttentionMetadata]] = {}
        self.fallback_logged = False
        self.replay_counts: dict[int, list[int]] = {}

    @staticmethod
    def capture_sizes(sizes: list[int], query_len: int, max_reqs: int) -> list[int]:
        limit = max(sizes, default=0)
        # Keep low-concurrency draft shapes independent of main-model rounding.
        exact = range(1, min(16, max_reqs, limit // query_len) + 1)
        return sorted(
            {reqs * query_len for reqs in exact}
            | {n for n in sizes if n % query_len == 0 and n // query_len <= max_reqs},
            reverse=True,
        )

    def _record_replay(self, query_len: int, hit: bool) -> None:
        counts = self.replay_counts.setdefault(query_len, [0, 0])
        counts[0 if hit else 1] += 1
        total = sum(counts)
        if total == 1 or total % 1000 == 0:
            logger.info(
                "MTP full graph query_len=%d: hits=%d fallbacks=%d",
                query_len,
                *counts,
            )

    def _forward(
        self,
        num_tokens: int,
        query_len: int,
        metadata: dict[str, TritonAttentionMetadata],
        mode: CUDAGraphMode,
    ) -> Any:
        p = self.proposer
        descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            num_reqs=num_tokens // query_len,
            uniform=query_len == 1,
        )
        kwargs = dict(
            input_ids=p.input_ids[:num_tokens],
            positions=p._get_positions(num_tokens),
            inputs_embeds=None,
            hidden_states=p.hidden_states[:num_tokens],
        )
        with set_forward_context(
            metadata,
            p.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=mode,
            batch_descriptor=descriptor,
            slot_mapping={name: md.slot_mapping for name, md in metadata.items()},
        ):
            return self.wrapper(**kwargs)

    def capture(self, block_table_width: int) -> None:
        p = self.proposer
        if len(p.draft_attn_groups) != 1:
            raise ValueError("Full MTP graphs require one draft attention group")
        group = p.draft_attn_groups[0]
        builder = group.get_metadata_builder()
        if group.backend.get_name() != "TRITON_ATTN":
            raise ValueError(
                "full_draft_cudagraph requires speculative attention_backend="
                "TRITON_ATTN"
            )
        sizes = p.compilation_config.cudagraph_capture_sizes or []
        max_reqs = p.max_batch_size
        max_seq_len = p.draft_model_config.max_model_len
        # Startup-only capture borrows runtime buffers. Every proposal populates
        # its active inputs before replay; captured KV stores use invalid slots.
        p.input_ids.fill_(1)
        p.positions.zero_()
        if p.uses_mrope:
            p.mrope_positions.zero_()
        p.hidden_states.zero_()
        for query_len in sorted({1, 1 + p.num_speculative_tokens}):
            for num_tokens in self.capture_sizes(sizes, query_len, max_reqs):
                num_reqs = num_tokens // query_len
                starts_cpu = torch.arange(
                    0, num_tokens + 1, query_len, dtype=torch.int32
                )
                common = CommonAttentionMetadata(
                    query_start_loc=starts_cpu.to(p.device),
                    query_start_loc_cpu=starts_cpu,
                    seq_lens=torch.full(
                        (num_reqs,), query_len, dtype=torch.int32, device=p.device
                    ),
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=query_len,
                    max_seq_len=max_seq_len,
                    block_table_tensor=torch.zeros(
                        (num_reqs, block_table_width),
                        dtype=torch.int32,
                        device=p.device,
                    ),
                    # Capture must not store draft keys/values into real pages.
                    slot_mapping=torch.full(
                        (num_tokens,), -1, dtype=torch.int64, device=p.device
                    ),
                )
                md = builder.build(0, common)
                assert isinstance(md, TritonAttentionMetadata)
                layer_md = {name: md for name in group.layer_names}
                # Lazy backend initialization must happen outside capture.
                self._forward(num_tokens, query_len, layer_md, CUDAGraphMode.NONE)
                self._forward(num_tokens, query_len, layer_md, CUDAGraphMode.FULL)
                self.metadata[num_tokens, query_len] = layer_md
        logger.info("Captured %d full MTP forward graphs", len(self.metadata))

    def replay(
        self,
        common: CommonAttentionMetadata,
        layer_metadata: dict[str, Any],
        num_tokens: int,
    ) -> Any | None:
        query_len = common.max_query_len
        actual = common.num_actual_tokens
        candidates = [n for n, q in self.metadata if q == query_len and n >= actual]
        graph_tokens = min(candidates, default=0)
        captured = self.metadata.get((graph_tokens, query_len))
        if common.query_start_loc_cpu is None:
            self._record_replay(query_len, False)
            return None
        starts = common.query_start_loc_cpu[: common.num_reqs + 1].numpy()
        uniform = (
            common.num_reqs > 0
            and actual <= num_tokens
            and common.num_reqs * query_len == actual
            and starts.size == common.num_reqs + 1
            and starts[0] == 0
            and np.all(np.diff(starts) == query_len)
        )
        if captured is None or not uniform:
            self._record_replay(query_len, False)
            if not self.fallback_logged:
                logger.info("MTP full graph fallback: uncaptured or non-uniform shape")
                self.fallback_logged = True
            return None
        if set(layer_metadata) != set(captured):
            self._record_replay(query_len, False)
            return None
        for name, source in layer_metadata.items():
            target = captured[name]
            if not isinstance(source, TritonAttentionMetadata):
                self._record_replay(query_len, False)
                return None
            # Optional attention features need their own capture contracts.
            if (
                source.use_cascade
                or source.causal is not True
                or source.mm_prefix_range_tensor is not None
                or source.rswa_prefix_lens is not None
                or source.scheduler_metadata is not None
                or source.prefix_scheduler_metadata is not None
            ):
                self._record_replay(query_len, False)
                return None
            if (
                source.block_table.shape[1:] != target.block_table.shape[1:]
                or source.block_table.shape[0] < common.num_reqs
                or source.max_seq_len > target.max_seq_len
            ):
                self._record_replay(query_len, False)
                return None
            if any(
                getattr(source, field).shape[0] < needed
                for field, needed in (
                    ("query_start_loc", common.num_reqs + 1),
                    ("seq_lens", common.num_reqs),
                    ("slot_mapping", actual),
                )
            ):
                self._record_replay(query_len, False)
                return None
        for name, source in layer_metadata.items():
            target = captured[name]
            # Layers in this group share metadata; stage it only once.
            if name != next(iter(captured)):
                continue
            # Dummy queries attend no keys and must never write KV pages.
            # Uniform query offsets are fixed at capture time.
            if graph_tokens > actual:
                target.seq_lens[common.num_reqs :].zero_()
                target.block_table[common.num_reqs :].zero_()
                target.slot_mapping[actual:].fill_(-1)
            for field, length in (
                ("query_start_loc", common.num_reqs + 1),
                ("seq_lens", common.num_reqs),
                ("block_table", common.num_reqs),
                ("slot_mapping", actual),
            ):
                dst = getattr(target, field)
                src = getattr(source, field)
                dst[:length].copy_(src[:length], non_blocking=True)
        self._record_replay(query_len, True)
        return self._forward(graph_tokens, query_len, captured, CUDAGraphMode.FULL)
