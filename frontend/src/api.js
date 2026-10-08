let csrfToken = null;

export async function loadCsrf() {
  const response = await fetch('/api/auth/csrf', { credentials: 'include' });
  if (!response.ok) throw new Error('Could not establish a secure session.');
  const data = await response.json();
  csrfToken = data.csrf_token;
  return csrfToken;
}

export async function apiFetch(path, options = {}, retry = true) {
  const method = (options.method || 'GET').toUpperCase();
  const write = !['GET', 'HEAD', 'OPTIONS'].includes(method);
  if (write && !csrfToken) await loadCsrf();
  const headers = new Headers(options.headers || {});
  if (write) headers.set('X-CSRF-Token', csrfToken);
  const response = await fetch(path, { ...options, headers, credentials: 'include' });
  if (response.status === 401 && retry && csrfToken) {
    const refreshed = await fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include',
      headers: { 'X-CSRF-Token': csrfToken },
    });
    if (refreshed.ok) return apiFetch(path, options, false);
  }
  return response;
}

export async function responseData(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error?.message || data.detail || 'The request failed.');
  return data;
}

export function newId() {
  return crypto.randomUUID();
}
