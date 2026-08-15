import { useState } from "react";
import type { CompletionRecord } from "../api/types";
import { JsonView } from "./JsonView";
import { CopyBtn } from "./CopyBtn";

interface Props {
  completion: CompletionRecord;
}

function Panel({ title, value }: { title: string; value: any }) {
  return (
    <div>
      <div className="mb-1 text-xs font-medium text-slate-500">{title}</div>
      <JsonView value={value} collapsed={false} maxHeight="280px" />
    </div>
  );
}

export function CompletionDiff({ completion }: Props) {
  const [open, setOpen] = useState(false);

  const originalRequest = completion.original_request ?? {};
  const transformedRequest = completion.transformed_request ?? completion.request ?? {};
  const response = completion.response ?? {};

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs">{completion.completion_id}</span>
          <span className="text-xs text-slate-500">{completion.api_type ?? "?"}</span>
          <span className="text-xs text-slate-500">
            {completion.model_requested ?? "?"} → {completion.model_used ?? "?"}
          </span>
        </div>
        <span className="text-xs text-slate-500">
          {completion.timestamp ? new Date(completion.timestamp).toLocaleTimeString() : ""}
          {" "}
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="grid grid-cols-1 gap-3 border-t border-slate-200 p-3 md:grid-cols-2">
          <Panel title="Raw client request (original API format)" value={originalRequest} />
          <Panel title="Transformed request" value={transformedRequest} />
          <Panel title="Inference engine response" value={response} />
          <Panel title="Metadata" value={completion.metadata ?? {}} />
        </div>
      )}
    </div>
  );
}
