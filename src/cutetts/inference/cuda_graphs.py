# Copyright 2026 FormoSpeech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA-graph replay of small, stateless per-step functions."""

from __future__ import annotations

from collections.abc import Callable

import torch


class GraphedFunction:
    """Calls `fn(x)` by replaying a CUDA graph captured for x's shape and dtype.

    For functions that launch many small kernels per decode step and keep no
    state between calls (a captured graph replays the same kernels on a fixed
    input buffer). A graph is captured on the first call with each shape and
    kept; at most `max_graphs` shapes are graphed, later ones run eagerly.
    Inputs off CUDA, under autocast, or outside inference mode run eagerly.
    The returned tensor is a copy, so it outlives the next replay.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], max_graphs: int = 32):
        self.fn = fn
        self.max_graphs = max_graphs
        self._graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self._pool = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not x.is_cuda
            or not torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cuda")
        ):
            return self.fn(x)
        key = (tuple(x.shape), x.dtype, x.device)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= self.max_graphs:
                return self.fn(x)
            entry = self._graphs[key] = self._capture(x)
        graph, static_input, static_output = entry
        static_input.copy_(x)
        graph.replay()
        return static_output.clone()

    def _capture(self, x: torch.Tensor) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
        device = x.device
        static_input = x.clone()
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # warm up kernels and workspaces before capture
            for _ in range(2):
                self.fn(static_input)
        torch.cuda.current_stream(device).wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_output = self.fn(static_input)
        return graph, static_input, static_output
