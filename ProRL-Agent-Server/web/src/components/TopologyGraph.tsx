import type { TopologyPayload } from "../api/types";
import { StatusPill } from "./StatusPill";

interface Props {
  topology: TopologyPayload | undefined;
  isLoading?: boolean;
}

function WorkerBar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = max > 0 ? Math.min(1, value / max) : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-slate-600">{label}</span>
      <div className="relative h-2 flex-1 overflow-hidden rounded bg-slate-200">
        <div
          className="absolute inset-y-0 left-0 bg-blue-500"
          style={{ width: `${pct * 100}%` }}
        />
      </div>
      <span className="w-12 text-right text-slate-600 font-mono">
        {value}/{max}
      </span>
    </div>
  );
}

export function TopologyGraph({ topology, isLoading }: Props) {
  if (isLoading || !topology) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        Loading topology…
      </div>
    );
  }

  const { rollout, gateways } = topology;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <div className="text-sm text-slate-500">Rollout</div>
          <div className="font-mono text-sm">{rollout.url}</div>
        </div>
        <StatusPill status={rollout.reachable ? "RUNNING" : "ERROR"} />
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {gateways.map((gw) => {
          const metrics = gw.health?.metrics || {};
          const activeStatusCounts = gw.health?.active_status_counts || {};
          const activeSessions: any[] = gw.health?.active_sessions || [];
          return (
            <div key={gw.node_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-500">Gateway</div>
                  <div className="font-mono text-sm">{gw.node_id}</div>
                  <div className="text-xs text-slate-500">{gw.gateway_url}</div>
                </div>
                <StatusPill status={gw.reachable ? "RUNNING" : "ERROR"} />
              </div>

              <div className="mt-3 space-y-1">
                <WorkerBar
                  label="init"
                  value={metrics.init_inflight ?? 0}
                  max={gw.max_init_workers}
                />
                <WorkerBar
                  label="run"
                  value={metrics.run_inflight ?? 0}
                  max={gw.max_run_workers}
                />
                <WorkerBar
                  label="postrun"
                  value={metrics.postrun_inflight ?? 0}
                  max={gw.max_postrun_workers}
                />
              </div>

              <div className="mt-3 border-t border-slate-200 pt-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">{(gw.engine || "engine").toUpperCase()}</span>
                  <span className="font-mono">{gw.inference_base_url}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">model</span>
                  <span className="font-mono">{gw.model_served || "—"}</span>
                </div>
              </div>

              {Object.keys(activeStatusCounts).length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1 text-xs">
                  {Object.entries(activeStatusCounts).map(([status, count]) => (
                    <StatusPill key={status} status={status} className="!text-[10px]">
                      {status}={String(count)}
                    </StatusPill>
                  ))}
                </div>
              )}

              {activeSessions.length > 0 && (
                <div className="mt-2 text-xs">
                  <div className="text-slate-500">
                    Active sessions ({activeSessions.length})
                  </div>
                  <ul className="mt-1 space-y-1">
                    {activeSessions.slice(0, 5).map((s) => (
                      <li key={s.session_id} className="font-mono text-[11px] truncate">
                        {s.session_id.slice(0, 18)}… <span className="text-slate-500">{s.status}</span>
                      </li>
                    ))}
                    {activeSessions.length > 5 && (
                      <li className="text-slate-500">+ {activeSessions.length - 5} more</li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
