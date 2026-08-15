import { useEffect } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Dashboard } from "./routes/Dashboard";
import { TasksList } from "./routes/TasksList";
import { TaskDetail } from "./routes/TaskDetail";
import { SessionDetail } from "./routes/SessionDetail";
import { Compare } from "./routes/Compare";
import { subscribePolarEvents } from "./api/sse";

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        `rounded px-3 py-1 text-sm ${
          isActive ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"
        }`
      }
    >
      {label}
    </NavLink>
  );
}

function NavBar() {
  return (
    <nav className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-2">
        <Link to="/" className="flex items-center gap-2 text-slate-900">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ background: "linear-gradient(135deg, #2563eb, #16a34a)" }}
          />
          <span className="font-semibold">Polar Dashboard</span>
        </Link>
        <div className="flex items-center gap-1">
          <NavItem to="/" label="Dashboard" />
          <NavItem to="/tasks" label="Tasks" />
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  const location = useLocation();
  const queryClient = useQueryClient();

  // SSE → query cache invalidation
  useEffect(() => {
    const controller = new AbortController();
    subscribePolarEvents((event) => {
      const data = event.data || {};
      switch (event.type) {
        case "task.created":
        case "task.updated":
        case "task.completed":
          queryClient.invalidateQueries({ queryKey: ["tasks"] });
          if (data.task_id) {
            queryClient.invalidateQueries({ queryKey: ["task", data.task_id] });
          }
          break;
        case "session.state_changed":
          if (data.session_id) {
            queryClient.invalidateQueries({ queryKey: ["session", data.session_id] });
            queryClient.invalidateQueries({
              queryKey: ["session-completions", data.session_id],
            });
          }
          if (data.task_id) {
            queryClient.invalidateQueries({ queryKey: ["task", data.task_id] });
          }
          queryClient.invalidateQueries({ queryKey: ["topology"] });
          break;
        case "session.completion_added":
          if (data.session_id) {
            queryClient.invalidateQueries({
              queryKey: ["session-completions", data.session_id],
            });
          }
          break;
        case "ping":
        case "hello":
          break;
      }
    }, controller);
    return () => controller.abort();
  }, [queryClient]);

  return (
    <div className="flex min-h-full flex-col">
      <NavBar />
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/tasks" element={<TasksList />} />
          <Route path="/tasks/:taskId" element={<TaskDetail />} />
          <Route path="/sessions/:sessionId" element={<SessionDetail />} />
          <Route path="/compare" element={<Compare />} />
          <Route
            path="*"
            element={
              <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
                Not found. Path: <code>{location.pathname}</code>
              </div>
            }
          />
        </Routes>
      </main>
      <footer className="border-t border-slate-200 bg-white px-4 py-2 text-center text-xs text-slate-500">
        Polar Dashboard · local · read-only ·{" "}
        <a className="underline" href="/docs" target="_blank" rel="noreferrer">
          OpenAPI
        </a>
      </footer>
    </div>
  );
}
