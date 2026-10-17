// The floating overlay. Rendered into a Document Picture-in-Picture window through a portal, or
// into a corner of the setup tab where that API is missing. Motion here is CSS only (see
// index.css); ShinyText is the one rAF component and simply holds still if the tab is throttled.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { media, score, type Verdict } from "./api";
import GlitchText from "./reactbits/GlitchText";
import ShinyText from "./reactbits/ShinyText";

export type OverlayState =
  | { kind: "dormant" }
  | { kind: "watching"; reason?: string | null; windowNo: number; maxWindows: number;
      windowMs: number; windowStartedAt: number; startedAt: number; note?: string }
  | { kind: "judging"; startedAt: number }
  | { kind: "pass"; score: number }
  | { kind: "alert"; verdict: Verdict; score?: number; text: string; frames: string[];
      sessionId?: string };

export type OverlayAction = "verify" | "details" | "retry" | "dismiss";

// Size the floating window opens at. Chrome ignores anything under its minimum (about 240x100
// outer) and opens a 720x449 window instead; useFit then shrinks it to the content.
export const PIP_OPEN: [number, number] = [260, 72];
const MIN_OUTER = [240, 100]; // Chrome's smallest floating window, title bar included

// Re-render every `ms` using the overlay window's own timers, which are not throttled.
function useTick(win: Window, ms: number, on: boolean) {
  const [, set] = useState(0);
  useEffect(() => {
    if (!on) return;
    const id = win.setInterval(() => set((n) => n + 1), ms);
    return () => win.clearInterval(id);
  }, [win, ms, on]);
}

// Resize the floating window to its content whenever the content's size changes. The observer
// belongs to the overlay window, so it keeps firing while the setup tab is throttled.
function useFit(win: Window, on: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!on || !el) return;
    let retry = 0;
    const fit = () => {
      const r = el.getBoundingClientRect();
      // Chrome sometimes drops a request below its minimum instead of clamping it, so clamp here.
      const w = Math.max(MIN_OUTER[0], Math.ceil(r.width) + win.outerWidth - win.innerWidth);
      const h = Math.max(MIN_OUTER[1], Math.ceil(r.height) + win.outerHeight - win.innerHeight);
      if (Math.abs(win.outerWidth - w) < 2 && Math.abs(win.outerHeight - h) < 2) return;
      try {
        win.resizeTo(w, h);
      } catch {
        /* window closing */
      }
      // A resize issued while the window is still opening or moving can be lost; check again.
      win.clearTimeout(retry);
      retry = win.setTimeout(fit, 250);
    };
    const Observer = (win as Window & typeof globalThis).ResizeObserver ?? ResizeObserver;
    const ro = new Observer(fit);
    ro.observe(el);
    fit();
    return () => {
      ro.disconnect();
      win.clearTimeout(retry);
    };
  }, [win, on]);
  return ref;
}

interface Props {
  state: OverlayState;
  ready: boolean; // a source is being watched
  inline: boolean;
  win: Window;
  onAction: (a: OverlayAction) => void;
}

export function Overlay({ state, ready, inline, win, onAction }: Props) {
  const ticking = state.kind === "watching" || state.kind === "judging";
  useTick(win, 250, ticking);
  const fit = useFit(win, !inline);
  const body = <Body state={state} ready={ready} onAction={onAction} />;
  if (inline) {
    return (
      <div className="fixed right-4 bottom-4 z-50 overflow-hidden rounded-2xl border border-line bg-bg shadow-2xl shadow-black/60">
        {body}
      </div>
    );
  }
  // The window cannot shrink below Chrome's minimum; centre the content in any spare room.
  return (
    <div className="grid min-h-screen place-items-center bg-bg">
      <div ref={fit} className="w-max">
        {body}
      </div>
    </div>
  );
}

// Natural-size content for each state; the window is fitted around it.
function Body({ state, ready, onAction }: Omit<Props, "inline" | "win">) {
  if (state.kind === "dormant") {
    return (
      <button
        onClick={() => onAction("verify")}
        title={ready ? "Watch the caller" : "Pick a source in the setup tab"}
        className="flex cursor-pointer items-center gap-2.5 px-4 py-3 font-mono text-xs text-muted"
      >
        <span className="relative grid size-3 place-items-center">
          {ready && <span className="absolute size-3 rounded-full bg-accent animate-ping-slow" />}
          <span className={`size-3 rounded-full ${ready ? "bg-accent" : "bg-[#3a3f4a]"}`} />
        </span>
        {ready ? "Liveness · ready" : "Liveness · no source"}
      </button>
    );
  }

  const frame = "flex flex-col gap-1.5 bg-bg px-4 py-3 font-mono";

  if (state.kind === "watching") {
    const secs = Math.round((Date.now() - state.startedAt) / 1000);
    return (
      <div className={`${frame} w-[320px]`}>
        {state.reason && (
          <div className="truncate text-[11px] uppercase tracking-widest text-warn">
            {state.reason}
          </div>
        )}
        <ShinyText
          text="Watching the caller…"
          speed={1.4}
          color="#6b7280"
          shineColor="#5eead4"
          className="text-[17px] font-extrabold"
        />
        <div className={`text-[12px] ${state.note?.includes("inconclusive") ? "text-warn" : "text-muted"}`}>
          {state.note?.includes("inconclusive")
            ? "Not enough movement yet: turn your head further, left then right."
            : "Keep talking. Turn your head slowly to each side."}
        </div>
        <div className="h-[3px] overflow-hidden rounded bg-line">
          <i
            key={state.windowStartedAt}
            className="block h-full animate-fill bg-accent"
            style={{ "--fill-ms": `${state.windowMs}ms` } as React.CSSProperties}
          />
        </div>
        <div className="flex justify-between text-[11px] text-muted">
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-bad animate-blink" />
            window {state.windowNo}/{state.maxWindows}
            {state.note ? ` · ${state.note}` : ""}
          </span>
          <span>{secs} s</span>
        </div>
      </div>
    );
  }

  if (state.kind === "judging") {
    return (
      <div className={`${frame} w-[280px]`}>
        <ShinyText
          text="Checking…"
          speed={1.4}
          color="#6b7280"
          shineColor="#5eead4"
          className="text-[17px] font-extrabold"
        />
        <div className="flex justify-between text-[11px] text-muted">
          <span>YOLO pose → NVIDIA Cosmos → W&amp;B agent</span>
          <span>{Math.round((Date.now() - state.startedAt) / 1000)} s</span>
        </div>
      </div>
    );
  }

  if (state.kind === "pass") {
    return (
      <div className={`${frame} flex-row items-center gap-3 px-5`}>
        <span className="text-2xl font-extrabold text-ok animate-pop">✓ LIVE</span>
        <span className="text-lg text-ok/80 animate-pop">{score(state.score)}</span>
      </div>
    );
  }

  const failed = state.verdict === "FAIL";
  const title = failed
    ? "NOT VERIFIED"
    : state.verdict === "INCONCLUSIVE"
      ? "COULD NOT CLEAR"
      : "CHECK FAILED";
  const pick =
    state.frames.length > 3
      ? [state.frames[0], state.frames[state.frames.length >> 1], state.frames.at(-1)!]
      : state.frames;
  return (
    <div className={`${frame} w-[380px] border-t-[3px] ${failed ? "border-bad" : "border-warn"}`}>
      <div className="flex items-baseline justify-between">
        <GlitchText
          speed={0.6}
          className={`mx-0! whitespace-nowrap text-[20px] font-extrabold ${failed ? "text-bad!" : "text-warn!"}`}
        >
          {title}
        </GlitchText>
        {state.score !== undefined && (
          <span className="text-sm text-muted">{score(state.score)}</span>
        )}
      </div>
      <p className="m-0 line-clamp-3 text-xs text-[#c9ccd3]">{state.text}</p>
      {pick.length > 0 && (
        <div className="flex gap-1">
          {pick.map((u) => (
            <img
              key={u}
              src={media(u)}
              onError={(e) => (e.currentTarget.style.display = "none")}
              className="aspect-[4/3] w-1/3 rounded object-cover"
            />
          ))}
        </div>
      )}
      <div className="mt-1 flex gap-1.5">
        {state.sessionId && <Btn onClick={() => onAction("details")}>Details</Btn>}
        <Btn onClick={() => onAction("retry")}>Watch again</Btn>
        <Btn onClick={() => onAction("dismiss")}>Dismiss</Btn>
      </div>
    </div>
  );
}

function Btn(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="cursor-pointer rounded-md border border-line bg-panel px-2.5 py-1 text-xs text-ink hover:border-accent"
    />
  );
}
