export const API_BASE = "http://localhost:8000";

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function postFormBlob(
  path: string,
  form: FormData,
  opts?: { expectedContentType?: string }
): Promise<Blob> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }

  const blob = await res.blob();

  const expected = opts?.expectedContentType;
  if (expected) {
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (ct && !ct.includes(expected.toLowerCase())) {
      console.warn(`Unexpected content-type: '${ct}', expected to include '${expected}'`);
    }
  }

  return blob;
}
