# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from molt.trainer.rollout.router import VllmRouterActor


def test_vllm_router_close_reaps_subprocess():
    events = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append(("wait", timeout))

    actor_cls = getattr(VllmRouterActor, "__ray_actor_class__", VllmRouterActor)
    actor = object.__new__(actor_cls)
    actor._proc = Process()

    actor.close()

    assert events == ["terminate", ("wait", 10)]
