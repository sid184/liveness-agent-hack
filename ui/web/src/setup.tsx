// Setup tab: pick what to watch, float the overlay, then forget about it. Owns the watch flow:
// the caller is never asked to do anything. The call is recorded in back-to-back windows; each
// one is POSTed as soon as it ends while the next records, until the server decides the call.

import { StrictMode, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { createRoot } from "react-dom/client";
import {
  API,
  evidenceFrames,
  getJSON,
  type Summary,
  type Watch,
  type WindowResult,
} from "./api";
import "./index.css";
import { Overlay, PIP_OPEN, type OverlayAction, type OverlayState } from "./overlay";
import DecryptedText from "./reactbits/DecryptedText";
import FaultyTerminal from "./reactbits/FaultyTerminal";
import StarBorder from "./reactbits/StarBorder";
import { SessionList } from "./SessionList";
import { Panel, PanelTitle } from "./Panel";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { ComputerVideoCallIcon, UserIcon, WebcamIcon } from "@hugeicons/core-free-icons";

declare global {
  interface Window {
    documentPictureInPicture?: {
      requestWindow(o: { width: number; height: number }): Promise<Window>;
    };
  }
}

// Copy this page's styles into the floating window. Same-origin sheets go over as text;
// cross-origin ones (Google Fonts) cannot be read, so their <link> is cloned instead.
function copyStyles(target: Document) {
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      const style = target.createElement("style");
      style.textContent = Array.from(sheet.cssRules, (r) => r.cssText).join("\n");
      target.head.append(style);
    } catch {
      if (sheet.ownerNode) target.head.append(sheet.ownerNode.cloneNode(true));
    }
  }
}

function recordClip(stream: MediaStream, ms: number, win: Window): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"]
      .find((m) => MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => resolve(new Blob(chunks, { type: rec.mimeType || "video/webm" }));
    rec.onerror = () => reject(new Error("recording failed"));
    rec.start(250);
    win.setTimeout(() => rec.state !== "inactive" && rec.stop(), ms);
  });
}

function chime(audio: AudioContext | null) {
  if (!audio) return;
  const o = audio.createOscillator();
  const g = audio.createGain();
  o.frequency.value = 520;
  g.gain.setValueAtTime(0.0001, audio.currentTime);
  g.gain.exponentialRampToValueAtTime(0.15, audio.currentTime + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.6);
  o.connect(g).connect(audio.destination);
  o.start();
  o.stop(audio.currentTime + 0.65);
}

function App() {
  const [source, setSource] = useState<"screen" | "camera" | null>(null);
  const [pip, setPip] = useState<Window | null>(null);
  const [active, setActive] = useState(false); // overlay started (floating or inline)
  const [ov, setOvRaw] = useState<OverlayState>({ kind: "dormant" });
  const [label, setLabel] = useState("");
  const [status, setStatus] = useState("");
  const [recent, setRecent] = useState<Summary[]>([]);

  const stream = useRef<MediaStream | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const audio = useRef<AudioContext | null>(null);
  const busy = useRef(false);
  const seq = useRef(0); // guards the delayed PASS collapse against newer states
  const gen = useRef(0); // bumped by End session; a check from an older session drops out
  const pipRef = useRef<Window | null>(null);
  const labelRef = useRef(label);
  labelRef.current = label;

  // Timers run in the overlay window: the setup tab is usually hidden behind the call and
  // background tabs have their timers throttled.
  const W = () => (pipRef.current && !pipRef.current.closed ? pipRef.current : window);
  const sleep = (ms: number) => new Promise((r) => W().setTimeout(r, ms));
  const setOv = (s: OverlayState) => {
    seq.current += 1;
    setOvRaw(s);
    return seq.current;
  };

  const loadRecent = () => getJSON<Summary[]>("/api/sessions?limit=8").then(setRecent, () => {});
  useEffect(() => void loadRecent(), []);

  async function pickSource(kind: "screen" | "camera") {
    try {
      const s =
        kind === "screen"
          ? await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 15 }, audio: false })
          : await navigator.mediaDevices.getUserMedia({
              video: { width: 1280, height: 720 },
              audio: false,
            });
      stream.current?.getTracks().forEach((t) => t.stop());
      stream.current = s;
      if (video.current) video.current.srcObject = s;
      s.getVideoTracks()[0].addEventListener("ended", () => {
        stream.current = null;
        setSource(null);
        setStatus("The shared source ended. Pick it again.");
        setOv({ kind: "dormant" });
      });
      setSource(kind);
      setStatus(kind === "screen" ? "Watching the shared window." : "Watching the webcam.");
    } catch (e) {
      setStatus(`No source: ${(e as Error).message || e}`);
    }
  }

  async function startOverlay() {
    audio.current ??= new AudioContext();
    if (active && (!pipRef.current || !pipRef.current.closed)) return;
    setOv({ kind: "dormant" });
    if (window.documentPictureInPicture) {
      const [width, height] = PIP_OPEN;
      const win = await window.documentPictureInPicture.requestWindow({ width, height });
      win.document.title = "Liveness";
      copyStyles(win.document);
      win.document.body.style.margin = "0";
      const g = gen.current;
      win.addEventListener("pagehide", () => {
        if (g !== gen.current) return; // closed by End session, which already cleaned up
        pipRef.current = null;
        setPip(null);
        setActive(false);
        setStatus("Overlay closed. Start it again to keep checking.");
      });
      pipRef.current = win;
      setPip(win);
    } else {
      setStatus("This browser has no floating windows; the overlay sits in this tab's corner.");
    }
    setActive(true);
    watch(); // the agent starts watching the call straight away; nobody asks the caller anything
  }

  // Stop watching, close the dot, and drop any check in flight.
  function endSession() {
    gen.current += 1;
    busy.current = false;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    const win = pipRef.current;
    pipRef.current = null;
    win?.close();
    setPip(null);
    setActive(false);
    setSource(null);
    setOv({ kind: "dormant" });
    setStatus("Session ended. Pick a source to start a new one.");
    loadRecent();
  }

  const watch = useCallback(async (reason?: string | null) => {
    if (busy.current) return;
    if (!stream.current) {
      setStatus("Pick a source first.");
      return;
    }
    busy.current = true;
    const g = gen.current;
    const ended = () => g !== gen.current;
    let decided: WindowResult | null = null;
    let failed: Error | null = null;
    try {
      const w = await getJSON<Watch>("/api/watch/start", { method: "POST" });
      const windowMs = w.window_s * 1000;
      const startedAt = Date.now();
      let note = "";

      const show = (windowNo: number, windowStartedAt: number) =>
        setOv({ kind: "watching", reason, windowNo, maxWindows: w.max_windows, windowMs,
                windowStartedAt, startedAt, note });

      // Judge a window while the next one records. The server takes them in order.
      const post = async (clip: Blob) => {
        const form = new FormData();
        form.append("call_id", w.call_id);
        form.append("clip", clip, "clip");
        if (labelRef.current) form.append("label", labelRef.current);
        try {
          const out = await getJSON<WindowResult>("/api/watch/window", {
            method: "POST",
            body: form,
          });
          note = `last ${out.verdict.toLowerCase()}`;
          if (out.decision !== "WATCHING") decided ??= out;
        } catch (e) {
          if (!decided) failed ??= e as Error; // a 409 after the decision is expected
        }
      };

      const posts: Promise<void>[] = [];
      for (let i = 1; i <= w.max_windows && !decided && !failed; i++) {
        if (ended() || !stream.current) return;
        show(i, Date.now());
        const clip = await recordClip(stream.current, windowMs, W());
        if (ended()) return;
        posts.push(post(clip));
      }
      if (!decided && !failed) setOv({ kind: "judging", startedAt: Date.now() });
      await Promise.all(posts);
      if (ended()) return;
      if (failed && !decided) throw failed;
      const out = decided as WindowResult | null;
      if (!out) throw new Error("the call ended without a decision");

      if (out.decision === "PASS") {
        const mine = setOv({ kind: "pass", score: out.score });
        await sleep(4000);
        if (mine === seq.current && !ended()) setOv({ kind: "dormant" });
      } else {
        setOv({ kind: "alert", verdict: out.verdict, score: out.score, text: out.reasoning,
                frames: evidenceFrames(out), sessionId: out.session_id });
        chime(audio.current);
      }
      loadRecent();
    } catch (e) {
      if (ended()) return;
      setOv({ kind: "alert", verdict: "ERROR", text: (e as Error).message || String(e),
              frames: [] });
      chime(audio.current);
    } finally {
      if (!ended()) busy.current = false;
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const es = new EventSource(`${API}/api/events`);
    es.addEventListener("message", (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "trigger") watch(ev.reason);
    });
    return () => es.close();
  }, [watch]);

  function onAction(a: OverlayAction) {
    if (a === "verify") watch();
    if (a === "details" && ov.kind === "alert") window.open(`/session/${ov.sessionId}`, "_blank");
    if (a === "retry") {
      setOv({ kind: "dormant" });
      watch();
    }
    if (a === "dismiss") setOv({ kind: "dormant" });
  }

  const overlay = active && (
    <Overlay state={ov} ready={!!source} inline={!pip} win={pip ?? window} onAction={onAction} />
  );

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="fixed inset-0">
        <FaultyTerminal
          tint="#5eead4"
          brightness={0.55}
          scale={1.6}
          timeScale={0.4}
          scanlineIntensity={0.5}
          curvature={0.15}
          mouseStrength={0.3}
        />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(7,8,11,.55),rgba(7,8,11,.95)_70%)]" />
      </div>

      <main className="relative mx-auto max-w-3xl px-4 py-16 sm:py-24">
        <h1 className="m-0 font-mono text-4xl font-extrabold leading-none sm:text-6xl">
          <DecryptedText
            text="Is this caller real?"
            animateOn="view"
            sequential
            speed={45}
            encryptedClassName="text-accent/60"
          />
        </h1>
        <p className="mt-4 mb-10 max-w-xl text-base text-muted">
          Set this up once per call, then forget about it. The agent watches the caller as the
          call goes on, never asks them to do anything, and decides on its own. A small dot
          floats over your call; only a failed check stays on screen.
        </p>

        <div className="grid gap-4 sm:grid-cols-2">
          <Panel className="flex flex-col">
            <PanelTitle aside={source && <LiveBadge />}>
              <StepNo n={1} done={!!source} /> What to watch
            </PanelTitle>
            <div className="grid grid-cols-2 gap-2">
              <Tile
                icon={ComputerVideoCallIcon}
                title="Call window"
                sub="Share the Meet or Zoom tab"
                on={source === "screen"}
                onClick={() => pickSource("screen")}
              />
              <Tile
                icon={WebcamIcon}
                title="Webcam"
                sub="This laptop's camera"
                on={source === "camera"}
                onClick={() => pickSource("camera")}
              />
            </div>
            <div className="relative mt-3 aspect-video overflow-hidden rounded-xl border border-line bg-black/60">
              <video
                ref={video}
                autoPlay
                muted
                playsInline
                className={`size-full object-cover ${source ? "" : "invisible"}`}
              />
              {!source && (
                <div className="absolute inset-0 grid place-items-center text-center font-mono text-[11px] uppercase tracking-widest text-muted/70">
                  No source yet
                </div>
              )}
            </div>
            <p className="mt-auto mb-0 pt-3 text-xs text-muted">
              Pin or spotlight the caller in the call app so their video fills the window.
            </p>
          </Panel>

          <Panel className="flex flex-col">
            <PanelTitle aside={active && <span className="text-accent">running</span>}>
              <StepNo n={2} done={active} /> Float the overlay
            </PanelTitle>
            <StarBorder
              as="button"
              color="#5eead4"
              disabled={!source || active}
              onClick={() => startOverlay().catch((e) => setStatus(String(e)))}
              className={`w-full font-mono text-base font-bold ${
                source && !active ? "cursor-pointer" : "cursor-default opacity-50"
              }`}
            >
              {active ? "Overlay is floating" : source ? "Start overlay" : "Pick a source first"}
            </StarBorder>
            {(source || active) && (
              <button
                onClick={endSession}
                className="mt-2 w-full cursor-pointer rounded-xl border border-bad/40 bg-bad/5 py-2 font-mono text-sm text-bad transition-colors hover:border-bad hover:bg-bad/15"
              >
                End session
              </button>
            )}

            <div className="relative mt-3 aspect-video overflow-hidden rounded-xl border border-line bg-[radial-gradient(ellipse_at_50%_70%,#1a2128,#0b0e12)]">
              <HugeiconsIcon
                icon={UserIcon}
                size={72}
                strokeWidth={1}
                className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-2 text-[#2a3440]"
              />
              <span className="absolute bottom-2 left-2 rounded bg-black/50 px-1.5 py-0.5 font-mono text-[10px] text-muted">
                Your call
              </span>
              <span className="absolute top-2.5 right-2.5 grid size-7 place-items-center rounded-full border border-line bg-bg/90">
                {active && <span className="absolute size-2.5 rounded-full bg-accent animate-ping-slow" />}
                <span className={`size-2.5 rounded-full ${active ? "bg-accent" : "bg-[#3a3f4a]"}`} />
              </span>
            </div>
            <ol className="mt-3 mb-5 flex list-none items-center justify-between gap-1 p-0 font-mono text-[11px] text-muted">
              <Flow dot="bg-accent" label="Dot waits" />
              <Arrow />
              <Flow dot="bg-warn" label="Agent watches" />
              <Arrow />
              <Flow dot="bg-bad" label="Only FAIL stays" />
            </ol>

            <div className="mt-auto">
              <div className="mb-2 font-mono text-[11px] uppercase tracking-widest text-muted">
                Ground truth <span className="normal-case tracking-normal">(eval only)</span>
              </div>
              <div className="grid grid-cols-3 rounded-lg border border-line bg-black/30 p-0.5">
                {[
                  ["", "none"],
                  ["live", "live"],
                  ["attack", "attack"],
                ].map(([v, t]) => (
                  <button
                    key={v}
                    onClick={() => setLabel(v)}
                    className={`cursor-pointer rounded-md py-1.5 font-mono text-xs transition-colors ${
                      label === v ? "bg-accent/15 text-accent" : "text-muted hover:text-ink"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <p className="mt-3 mb-0 text-xs text-muted">
                Watching starts with the overlay. Click the dot to watch again, or start it from
                another system with{" "}
                <code className="rounded bg-black/40 px-1 font-mono text-accent/80">
                  POST /api/trigger
                </code>
                .
              </p>
            </div>
          </Panel>
        </div>

        <div className="mt-3 min-h-5 font-mono text-xs text-muted">{status}</div>

        <Panel className="mt-6">
          <PanelTitle aside={recent.length > 0 && <span>{recent.length}</span>}>
            Recent verdicts
          </PanelTitle>
          {recent.length ? (
            <SessionList items={recent} maxHeight={320} />
          ) : (
            <p className="m-0 text-sm text-muted">
              No checks yet. Verdicts show up here, each linked to its full review.
            </p>
          )}
        </Panel>
      </main>

      {overlay && (pip ? createPortal(overlay, pip.document.body) : overlay)}
    </div>
  );
}

function StepNo({ n, done }: { n: number; done: boolean }) {
  return (
    <span
      className={`grid size-5 place-items-center rounded-full text-[10px] ${
        done ? "bg-accent text-bg" : "border border-accent/40 text-accent"
      }`}
    >
      {done ? "✓" : n}
    </span>
  );
}

function LiveBadge() {
  return (
    <span className="flex items-center gap-1.5 text-bad">
      <span className="size-1.5 rounded-full bg-bad animate-blink" /> live
    </span>
  );
}

function Tile({ icon, title, sub, on, onClick }: {
  icon: IconSvgElement; title: string; sub: string; on: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`group flex cursor-pointer flex-col items-start gap-2 rounded-xl border p-3 text-left transition-all ${
        on
          ? "border-accent bg-accent/10 shadow-[0_0_24px_-8px_rgba(94,234,212,.6)]"
          : "border-line bg-black/30 hover:-translate-y-0.5 hover:border-accent/50"
      }`}
    >
      <HugeiconsIcon
        icon={icon}
        size={22}
        className={on ? "text-accent" : "text-muted group-hover:text-ink"}
      />
      <span>
        <span className="block text-sm font-semibold text-ink">{title}</span>
        <span className="block text-[11px] text-muted">{sub}</span>
      </span>
    </button>
  );
}

function Flow({ dot, label }: { dot: string; label: string }) {
  return (
    <li className="flex flex-col items-center gap-1.5 text-center">
      <span className={`size-2.5 rounded-full ${dot}`} />
      {label}
    </li>
  );
}

function Arrow() {
  return <li aria-hidden className="mb-5 h-px flex-1 bg-gradient-to-r from-line to-accent/40" />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
