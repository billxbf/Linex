# Polar Dashboard

A read-only **observability dashboard** for a running Polar stack: a React SPA
plus a single FastAPI service that proxies (read-only) to the rollout server and
gateway nodes and reads finished tasks straight off disk.

Launch it with `polar dashboard -c topology.yaml [--port 8090]` (see the
[top-level README](../../../README.md#cli-interface)). Defaults: binds
`127.0.0.1:8090`; the rollout URL and `save_dir` come from the topology (override
with `--rollout-url` / `--save-dir`).

## Mental model

The dashboard reads from **two sources and merges them**:

- **Live HTTP** to the rollout server and each gateway (current tasks, sessions,
  completions) via `UpstreamClient`.
- **On disk** under `save_dir` (finished tasks/sessions) via `FsIndex`, which
  scans `task_*/` every couple of seconds.

For live updates it runs an **SSE fan-out**: `SseFanout` opens one `/events`
stream per upstream (rollout + each gateway), tags each event with its source,
and republishes onto an in-process `EventBus` that the browser consumes at
`GET /api/events`. Everything is read-only; the only state-changing call is the
cancel proxy.

## Main files

- `cli.py`: registers the `dashboard` subcommand and starts the server.
- `config.py`: resolves host/port/rollout-url/save-dir from the topology + CLI
  overrides.
- `server.py`: builds the FastAPI app, starts the upstream clients + fs poller +
  SSE fan-outs, and serves the bundled SPA.
- `api/`: the HTTP route handlers (`topology`, `tasks`, `sessions`, `events`) —
  where the `/api/*` endpoints below actually live.
- `fs_index.py`: the on-disk index of `task_*` dirs (`TaskSummary` /
  `SessionSummary`).
- `sse_fanout.py` + `events.py`: upstream `/events` subscription and the
  in-process event bus.
- `upstream.py`: the async httpx client used to read rollout + gateways.

## Frontend build

The wheel ships a real UI only if `web/dist/` is built; without it the service
serves a small JSON placeholder.

```bash
cd web && npm install && npm run build      # writes web/dist/
```

Dev loop with hot reload (runs at <http://127.0.0.1:5173/>, proxies `/api/*` to
`:8090`):

```bash
cd web && npm install && npm run dev
```

## API surface (under `/api`)

| Path | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Service health + upstream reachability |
| `/api/topology` | GET | Static topology + live `/health` per gateway |
| `/api/tasks` | GET | List tasks (filesystem + live rollout overlay) |
| `/api/tasks/{id}` | GET | Single task + session summaries |
| `/api/sessions/{id}` | GET | Session detail |
| `/api/sessions/{id}/trajectory` | GET | Built trajectory traces |
| `/api/sessions/{id}/completions` | GET | Completion records (gateway, then disk) |
| `/api/sessions/{id}/evaluation` | GET | Evaluator outcome / strategy |
| `/api/sessions/{id}/raw` | GET | Raw on-disk session payload |
| `/api/sessions/{id}` | DELETE | Cancel a running session |
| `/api/events` | GET (SSE) | Fan-out of rollout + gateway events |

## Read-only endpoints it depends on

The dashboard relies on a few read-only endpoints in the other services:

- Rollout: `GET /tasks`, `GET /tasks/{id}/sessions`, `GET /events` (SSE).
- Gateway: `GET /sessions`, `GET /sessions/{id}/completions`, `GET /events` (SSE).
- Gateway completion records persist to
  `<save_dir>/task_<task_id>/sessions/<sid>/completions/<NNNN>-<id>.json` via the
  `CompletionWriter` background task, controlled by
  `gateway.completion_persistence` in `topology.yaml`.

## Task submission

Submission stays in the usual channels — `polar submit`, the example scripts, or
any client posting to `POST /rollout/task/submit`. The dashboard surfaces a task
as soon as it appears in the rollout's memory or in `<save_dir>/`.
