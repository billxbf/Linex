async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const init: RequestInit = {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  };
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail: any;
    try {
      detail = await response.json();
    } catch {
      detail = await response.text();
    }
    const error = new Error(
      `HTTP ${response.status} ${response.statusText}: ${
        typeof detail === "string" ? detail : JSON.stringify(detail)
      }`,
    );
    (error as any).status = response.status;
    (error as any).detail = detail;
    throw error;
  }
  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return (await response.text()) as unknown as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body: unknown) => request<T>("POST", path, body),
  delete: <T>(path: string) => request<T>("DELETE", path),
};
