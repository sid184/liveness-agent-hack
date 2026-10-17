// Shapes returned by ui/server.py.

// Where the API lives. Empty when FastAPI serves these pages itself. When the pages are hosted on
// Vercel the build defaults it to http://localhost:8765 (the laptop running `make ui`); open the
// page once with ?api=<url> to point it elsewhere, e.g. a tunnel. The choice is kept per browser.
function apiBase(): string {
  const fromQuery = new URLSearchParams(location.search).get("api");
  try {
    if (fromQuery !== null) localStorage.setItem("api", fromQuery);
    return localStorage.getItem("api") ?? import.meta.env.VITE_API_BASE ?? "";
  } catch {
    return fromQuery ?? import.meta.env.VITE_API_BASE ?? "";
  }
}
export const API = apiBase().replace(/\/$/, "");
export const media = (u: string) => `${API || location.origin}${u}`;

export type Verdict = "PASS" | "FAIL" | "INCONCLUSIVE" | "ERROR";
export type Decision = Verdict | "WATCHING";

export interface Summary {
  session_id: string;
  challenge_id: string;
  verdict: Verdict;
  score: number;
  instruction: string;
  reasoning: string;
  created_at: number;
  label: string | null;
}

export interface Watch {
  call_id: string;
  window_s: number;
  max_windows: number;
}

// One stage of the pipeline as run_session recorded it: what ran, where, how long.
export interface Stage {
  step: string;
  component: string;
  latency_s: number | null;
  detail?: string;
}

export interface WindowResult extends Session {
  call_id: string;
  decision: Decision;
  window_no: number;
  max_windows: number;
}

export interface Session extends Summary {
  frames: string[];
  clip: string | null;
  cited: number[]; // 1-based frame numbers Cosmos named as evidence
  stack: Stage[];
  similar: Summary[];
  signals?: {
    vlm_raw?: string;
    vlm_reasoning?: string | null;
    vlm_model?: string;
    vlm_latency_s?: number;
    prompt_version?: string;
    order?: string[];
    geometric?: {
      gates?: Record<string, boolean>;
      sequence?: string[];
      geo?: number;
      yaw_range?: number;
      pitch_range?: number;
    };
    call_id?: string | null;
    window_no?: number;
    window_verdict?: Verdict;
    call?: {
      decision: Verdict;
      windows: { session_id: string; verdict: Verdict; score: number }[];
      agent?: { text?: string; model?: string; error?: string };
    };
    vlm?: { observed?: string; artifacts?: string | null; verdict?: string; confidence?: number };
    vss?: { object_key?: string; tags?: string[]; error?: string };
  };
}

// Up to three frames for the overlay: the ones Cosmos cited, else first, middle and last.
export function evidenceFrames(s: Pick<Session, "frames" | "cited">): string[] {
  const cited = (s.cited ?? []).map((i) => s.frames[i - 1]).filter(Boolean);
  if (cited.length) return cited.slice(0, 3);
  const f = s.frames;
  return f.length > 3 ? [f[0], f[f.length >> 1], f[f.length - 1]] : f;
}

export const when = (t: number) => new Date(t * 1000).toLocaleString();
export const score = (s: number) => Number(s).toFixed(2);

export const VERDICT_COLOR: Record<Verdict, string> = {
  PASS: "text-ok",
  FAIL: "text-bad",
  INCONCLUSIVE: "text-warn",
  ERROR: "text-warn",
};

export async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${url}`, init);
  const body = await r.json();
  if (!r.ok) throw new Error(body.detail || `${url} ${r.status}`);
  return body as T;
}
