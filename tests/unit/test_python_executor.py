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

import importlib.util
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[2] / "examples" / "python" / "tools" / "python_executor.py"
_spec = importlib.util.spec_from_file_location("python_executor", _TOOL_PATH)
python_executor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(python_executor)
run_python = python_executor.run_python


def test_run_python_returns_stdout():
    assert run_python("print(6 * 7)").strip() == "42"


def test_run_python_reports_error_on_nonzero_exit():
    out = run_python("1 / 0")
    assert "ZeroDivisionError" in out


def test_run_python_isolates_file_writes_to_tempdir(tmp_path, monkeypatch):
    # cwd into a clean temp dir so a regression litters here (auto-cleaned by
    # pytest), not the repo root — and so we can assert nothing leaked into cwd.
    monkeypatch.chdir(tmp_path)
    sentinel = "py_exec_sentinel.txt"
    out = run_python(f"open({sentinel!r}, 'w').write('x'); print('done')")
    assert "done" in out
    # The snippet ran in its own throwaway cwd, so nothing lands in our cwd.
    assert not (tmp_path / sentinel).exists()
    assert list(tmp_path.iterdir()) == []
