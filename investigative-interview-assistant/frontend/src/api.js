const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export async function request(path, options = {}) {
  const headers = options.body instanceof FormData ? options.headers : {'Content-Type': 'application/json', ...options.headers};
  const response = await fetch(`${API}${path}`, {...options, headers});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || 'Request failed');
  return body;
}
export const wsUrl = id => `${API.replace(/^http/, 'ws')}/ws/interviews/${id}`;

