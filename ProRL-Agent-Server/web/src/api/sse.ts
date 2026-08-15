import { fetchEventSource } from "@microsoft/fetch-event-source";

export type PolarEvent = {
  type: string;
  ts: string;
  data: Record<string, any>;
};

export function subscribePolarEvents(
  onEvent: (event: PolarEvent) => void,
  controller: AbortController,
): void {
  fetchEventSource("/api/events", {
    signal: controller.signal,
    openWhenHidden: true,
    async onopen(response) {
      if (!response.ok) {
        throw new Error(`SSE open failed: ${response.status}`);
      }
    },
    onmessage(event) {
      if (!event.data) return;
      try {
        const parsed = JSON.parse(event.data) as PolarEvent;
        onEvent(parsed);
      } catch {
        // ignore malformed payloads
      }
    },
    onerror() {
      // The library auto-reconnects with backoff; swallow per-error noise.
    },
  }).catch(() => {
    // Ignore — the AbortController controls shutdown.
  });
}
