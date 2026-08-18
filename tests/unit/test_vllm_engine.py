# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace


def _install_vllm_test_stub():
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.21.0"

    @dataclass
    class AsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = object

    inputs = ModuleType("vllm.inputs")

    class TokensPrompt(dict):
        pass

    inputs.TokensPrompt = TokensPrompt

    utils = ModuleType("vllm.utils")
    utils.random_uuid = lambda: "test-request-id"

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = inputs
    sys.modules["vllm.utils"] = utils


try:
    import vllm.inputs  # noqa: F401
except Exception:
    _install_vllm_test_stub()

import molt.trainer.vllm.vllm_engine as vllm_engine  # noqa: E402


def test_vllm_ray_executor_uses_worker_gpu_even_when_actor_is_cpu_only():
    assert vllm_engine._vllm_worker_num_gpus("ray", 0) == 1
    assert vllm_engine._vllm_worker_num_gpus("mp", 8) == 8
    assert vllm_engine._vllm_worker_num_gpus("uni", 1) == 1


def test_format_ray_gpu_ids_keeps_all_visible_devices():
    assert vllm_engine._format_ray_gpu_ids([0.0, 2.0]) == "0,2"
    assert vllm_engine._format_ray_gpu_ids(["GPU-abc"]) == "GPU-abc"


def test_filter_vllm_engine_kwargs_drops_unsupported_optional_args(monkeypatch):
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {
            "model": "model-path",
            "dtype": "bfloat16",
            "gdn_prefill_backend": "triton",
        }
    )

    assert filtered == {"model": "model-path", "dtype": "bfloat16"}


def test_filter_vllm_engine_kwargs_keeps_speculative_config(monkeypatch):
    # MTP rollout passes speculative_config; it is a real AsyncEngineArgs field,
    # so it must survive the whitelist filter (not be dropped like unknown kwargs).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        speculative_config: object = None

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}
    )

    assert filtered == {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}


def test_filter_vllm_engine_kwargs_keeps_disable_custom_all_reduce(monkeypatch):
    # --vllm.disable_custom_all_reduce threads through as this kwarg; it is a real
    # AsyncEngineArgs field, so it must survive the whitelist filter (a rename/typo
    # would otherwise let it be silently dropped -> flag becomes a no-op).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        disable_custom_all_reduce: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs({"model": "m", "disable_custom_all_reduce": True})

    assert filtered == {"model": "m", "disable_custom_all_reduce": True}


def test_ray_visible_device_flag_is_cuda_only():
    assert vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})
    assert not vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_OTHER_VISIBLE_DEVICES": "1"})


def test_rollout_ray_actor_init_is_synchronous():
    """Ray actor constructors are synchronous; an async __init__ would not be awaited."""
    actor_cls = getattr(vllm_engine.RolloutRayActor, "__ray_actor_class__", vllm_engine.RolloutRayActor)
    assert not inspect.iscoroutinefunction(actor_cls.__init__)


def test_openai_server_enables_configured_output_parsers(monkeypatch):
    args = SimpleNamespace(enable_auto_tool_choice=False, tool_call_parser=None, reasoning_parser=None)
    app = SimpleNamespace(state=SimpleNamespace())
    api_server = SimpleNamespace(
        build_app=lambda *_: app,
        init_app_state=lambda *_: asyncio.sleep(0),
    )
    parser = SimpleNamespace(parse_args=lambda _: args)

    uvicorn = ModuleType("uvicorn")
    uvicorn.Config = lambda *_, **__: None
    uvicorn.Server = lambda _: SimpleNamespace(
        started=True,
        servers=[SimpleNamespace(sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 8000))])],
        serve=lambda: asyncio.sleep(0),
    )
    cli_args = ModuleType("vllm.entrypoints.openai.cli_args")
    cli_args.make_arg_parser = lambda _: parser
    argparse_utils = ModuleType("vllm.utils.argparse_utils")
    argparse_utils.FlexibleArgumentParser = object
    openai = ModuleType("vllm.entrypoints.openai")
    openai.api_server = api_server
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", ModuleType("vllm.entrypoints"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", openai)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.cli_args", cli_args)
    monkeypatch.setitem(sys.modules, "vllm.utils.argparse_utils", argparse_utils)

    actor_cls = getattr(vllm_engine.RolloutRayActor, "__ray_actor_class__", vllm_engine.RolloutRayActor)
    actor = actor_cls.__new__(actor_cls)
    actor.kwargs = {"model": "model-path"}
    actor.llm = SimpleNamespace(
        get_supported_tasks=lambda: asyncio.sleep(0, result=("generate",)),
        model_config=object(),
    )

    asyncio.run(actor.serve_openai(tool_call_parser="qwen3_coder", reasoning_parser="qwen3"))

    assert args.enable_auto_tool_choice is True
    assert args.tool_call_parser == "qwen3_coder"
    assert args.reasoning_parser == "qwen3"
