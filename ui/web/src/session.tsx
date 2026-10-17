// Review page for one window, or for the window that decided a call: everything the overlay
// leaves out.

import { StrictMode, useEffect, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { getJSON, media, VERDICT_COLOR, when, type Session, type Stage, type Summary } from "./api";
import "./index.css";
import CountUp from "./reactbits/CountUp";
import GlitchText from "./reactbits/GlitchText";
import GradientText from "./reactbits/GradientText";
import LetterGlitch from "./reactbits/LetterGlitch";
import { Panel, PanelTitle } from "./Panel";
import { SessionList } from "./SessionList";

function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Panel className="mb-4">
      <PanelTitle>{title}</PanelTitle>
      {children}
    </Panel>
  );
}

function Verdict({ s }: { s: Session }) {
  const size = "font-mono text-6xl font-extrabold leading-none sm:text-8xl";
  if (s.verdict === "PASS") {
    return (
      <GradientText colors={["#3ecf8e", "#5eead4", "#a7f3d0", "#3ecf8e"]} className={size}>
        PASS
      </GradientText>
    );
  }
  return (
    <GlitchText className={`mx-0! ${size} ${VERDICT_COLOR[s.verdict]}!`}>{s.verdict}</GlitchText>
  );
}

const pre = "m-0 whitespace-pre-wrap break-words font-mono text-xs text-[#c9ccd3]";

// Each stage of the pipeline with the component that ran it, so the stack is legible per verdict.
function Pipeline({ stack }: { stack: Stage[] }) {
  if (!stack.length) return <p className="m-0 text-sm text-muted">Not recorded for this session.</p>;
  return (
    <ol className="m-0 list-none p-0">
      {stack.map((st, i) => (
        <li key={i} className="grid grid-cols-[1.5rem_1fr_auto] gap-x-3 border-t border-line py-2 first:border-t-0">
          <span className="font-mono text-xs text-muted">{i + 1}</span>
          <div className="min-w-0">
            <div className="text-[15px]">
              {st.step} <span className="text-accent">· {st.component}</span>
            </div>
            {st.detail && <div className="break-words text-xs text-muted">{st.detail}</div>}
          </div>
          <span className="font-mono text-xs text-muted">
            {st.latency_s != null ? `${st.latency_s.toFixed(1)} s` : ""}
          </span>
        </li>
      ))}
    </ol>
  );
}

function Gates({ gates }: { gates?: Record<string, boolean> }) {
  if (!gates) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {Object.entries(gates).map(([k, v]) => (
        <span
          key={k}
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${v ? "border-ok/40 text-ok" : "border-bad/50 text-bad"}`}
        >
          {v ? "✓" : "✗"} {k.replace("_", " ")}
        </span>
      ))}
    </div>
  );
}

function App() {
  const id = location.pathname.split("/").pop()!;
  const [s, setS] = useState<Session | null | undefined>(undefined);
  const [history, setHistory] = useState<Summary[]>([]);

  useEffect(() => {
    getJSON<Session>(`/api/sessions/${id}`).then(setS, () => setS(null));
    getJSON<Summary[]>("/api/sessions").then(setHistory, () => {});
  }, [id]);

  return (
    <div className="relative min-h-screen">
      <div className="fixed inset-0 opacity-40">
        <LetterGlitch
          glitchColors={["#0f3b35", "#5eead4", "#ff4d5e"]}
          backgroundColor="#07080b"
          centerFade={0.9}
          edgeFade={0.2}
          speed={40}
        />
      </div>
      <main className="relative mx-auto max-w-4xl px-4 py-12">
        <a href="/" className="font-mono text-xs uppercase tracking-widest text-accent">
          ← Liveness agent
        </a>
        {s === undefined && <p className="text-muted">Loading…</p>}
        {s === null && <p className="text-muted">Session {id} not found.</p>}
        {s && (
          <>
            <header className="mt-8 mb-8 flex flex-wrap items-end gap-x-6 gap-y-2">
              <Verdict s={s} />
              <span className={`font-mono text-4xl font-bold ${VERDICT_COLOR[s.verdict]}`}>
                <CountUp to={s.score} decimals={2} duration={1.2} />
              </span>
              <p className="m-0 w-full text-muted">
                {s.signals?.call
                  ? `Call decided ${s.verdict}${s.signals.call.assurance ? `, ${s.signals.call.assurance} assurance,` : ""} after ${s.signals.call.windows.length} windows. The caller was never asked to do anything.`
                  : `One ${s.signals?.call_id ? "window of a call still being watched" : "recorded window"}; verdict ${s.verdict}.`}
                <br />
                <span className="font-mono text-xs">
                  {when(s.created_at)} · passive watch
                  {s.signals?.window_no ? ` · window ${s.signals.window_no}` : ""} · prompt{" "}
                  {s.signals?.prompt_version ?? "?"}
                  {s.label ? ` · test label ${s.label}` : ""}
                </span>
              </p>
            </header>

            <Card title="How this verdict was made">
              <Pipeline stack={s.stack ?? []} />
            </Card>
            {s.signals?.call && (
              <Card title={`Call decision · ${s.signals.call.windows.length} windows`}>
                <div className="flex flex-wrap gap-1.5">
                  {s.signals.call.windows.map((w, i) => (
                    <a
                      key={w.session_id}
                      href={`/session/${w.session_id}`}
                      className={`rounded border border-line px-2 py-0.5 font-mono text-xs no-underline ${VERDICT_COLOR[w.verdict]}`}
                    >
                      {i + 1}. {w.verdict.toLowerCase()} {w.score.toFixed(2)}
                      {w.assurance ? ` · ${w.assurance}` : ""}
                    </a>
                  ))}
                </div>
              </Card>
            )}
            <Card title={s.signals?.call?.agent?.text ? "Decision note · W&B agent" : "Reasoning"}>
              <p className="m-0 whitespace-pre-wrap text-[15px] leading-relaxed">{s.reasoning}</p>
            </Card>
            {s.clip && (
              <Card title="Recording">
                <video src={media(s.clip)} controls className="w-full rounded-lg bg-black" />
              </Card>
            )}
            <Card title="Physical coherence · NVIDIA Cosmos">
              {s.signals?.vlm?.observed && (
                <p className="mt-0 text-[15px] leading-relaxed">{s.signals.vlm.observed}</p>
              )}
              <p className={`mt-0 text-sm ${s.signals?.vlm?.artifacts ? "text-bad" : "text-muted"}`}>
                Artefacts: {s.signals?.vlm?.artifacts ?? "none named"}
              </p>
              <Gates gates={s.signals?.geometric?.gates} />
              {s.signals?.geometric?.cues && (
                <p className="mt-2 mb-0 text-xs text-muted">
                  YOLO: natural movement {s.signals.geometric.cues.pose_jitter?.toFixed(2)} face /{" "}
                  {s.signals.geometric.cues.head_body?.toFixed(2)} head vs shoulders; hard moments:{" "}
                  {s.signals.geometric.cues.stress_frames
                    ? `${s.signals.geometric.cues.stress_frames} frames (${(s.signals.geometric.cues.stress_kinds ?? []).join(", ")})`
                    : "none, the caller stayed still (basic assurance at most)"}
                  .
                </p>
              )}
              {s.frames.length > 0 && (
                <p className="mt-4 mb-1.5 text-xs text-muted">
                  The {s.frames.length} frames Cosmos saw, picked by YOLO: hard moments first (a
                  hand or cup at the face, the furthest turns), then spread. Red = cited as evidence.
                </p>
              )}
              {s.frames.length > 0 && (
                <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-1.5">
                  {s.frames.map((u, i) => {
                    const hit = s.cited.includes(i + 1);
                    return (
                      <figure
                        key={u}
                        className={`relative m-0 rounded-md ${hit ? "ring-2 ring-bad" : ""}`}
                      >
                        <img src={media(u)} className="w-full rounded-md" />
                        <figcaption
                          className={`absolute top-1 left-1 rounded px-1 font-mono text-[10px] ${hit ? "bg-bad text-black" : "bg-black/70"}`}
                        >
                          {i + 1}
                          {hit ? " cited" : ""}
                        </figcaption>
                        {s.signals?.frame_notes?.[i] && (
                          <figcaption className="absolute bottom-1 left-1 rounded bg-black/70 px-1 font-mono text-[10px]">
                            {s.signals.frame_notes[i]}
                          </figcaption>
                        )}
                      </figure>
                    );
                  })}
                </div>
              )}
            </Card>
            <Card title="Similar past windows · Cosmos Embed1 over the verdict log">
              <SessionList items={s.similar} maxHeight={260} />
            </Card>
            <Card title="Raw model output">
              {s.signals?.vlm_reasoning && s.signals.vlm_reasoning !== s.signals.vlm_raw && (
                <pre className={`${pre} mb-4 text-muted`}>{s.signals.vlm_reasoning}</pre>
              )}
              <pre className={pre}>{s.signals?.vlm_raw ?? ""}</pre>
            </Card>
            <Card title="Signals">
              <pre className={pre}>
                {JSON.stringify({ geometric: s.signals?.geometric, vlm: s.signals?.vlm }, null, 2)}
              </pre>
            </Card>
          </>
        )}
        <Card title="Recent calls">
          <SessionList items={history} />
        </Card>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
