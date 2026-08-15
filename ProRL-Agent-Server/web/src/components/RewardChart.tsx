interface Props {
  rewards: (number | null | undefined)[];
}

export function RewardChart({ rewards }: Props) {
  const valid = rewards.filter((r): r is number => typeof r === "number");
  if (valid.length === 0) {
    return <div className="text-sm text-slate-500">No rewards yet.</div>;
  }
  const min = Math.min(...valid, 0);
  const max = Math.max(...valid, 1);
  const range = max - min || 1;
  const mean = valid.reduce((a, b) => a + b, 0) / valid.length;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
        <span>
          n={valid.length} · min={min.toFixed(2)} · max={max.toFixed(2)} · mean=
          {mean.toFixed(2)}
        </span>
      </div>
      <div className="flex h-16 items-end gap-1">
        {rewards.map((r, i) => {
          if (r == null) {
            return (
              <div
                key={i}
                className="flex-1 border-b-2 border-dashed border-slate-300"
                title={`session ${i + 1}: n/a`}
              />
            );
          }
          const h = ((r - min) / range) * 100;
          const color =
            r >= 0.75 ? "bg-green-500" : r >= 0.4 ? "bg-amber-500" : "bg-red-500";
          return (
            <div
              key={i}
              className={`${color} flex-1 rounded-t`}
              style={{ height: `${Math.max(4, h)}%` }}
              title={`session ${i + 1}: ${r.toFixed(2)}`}
            />
          );
        })}
      </div>
    </div>
  );
}
