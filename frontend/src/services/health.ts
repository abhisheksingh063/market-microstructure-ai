const HEALTH_BASE = "/health";

async function request<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface HealthStatus {
  status: string;
}

export interface ApiHealth {
  status: string;
  version: string;
  environment: string;
}

export interface DatabaseHealth {
  status: string;
  database: string;
}

export const healthService = {
  get: () => request<HealthStatus>(HEALTH_BASE),
  getApi: () => request<ApiHealth>(`${HEALTH_BASE}/api`),
  getDatabase: () => request<DatabaseHealth>(`${HEALTH_BASE}/database`),
};
