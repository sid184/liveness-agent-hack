import { firstSentence, score, when, type Summary, type Verdict } from "./api";

// Past calls as a scoreboard: verdict colour on the edge, one dot per window, assurance pill.
const EDGE: Record<string, string> = {
  PASS: "border-l-ok",
  FAIL: "border-l-bad",
  INCONCLUSIVE: "border-l-warn",
  ERROR: "border-l-line",
};
const TEXT: Record<string, string> = {
  PASS: "text-ok",
  FAIL: "text-bad",
  INCONCLUSIVE: "text-warn",
  ERROR: "text-muted",
};
const DOT: Record<string, string> = {
  PASS: "bg-ok",
  FAIL: "bg-bad",
  INCONCLUSIVE: "bg-warn/70",
  ERROR: "bg-line",
};

function Dots({ verdicts, count }: { verdicts?: Verdict[] | null; count?: number | null }) {
  const list: (Verdict | null)[] = verdicts?.length
    ? verdicts
    : Array.from({ length: count ?? 1 }, () => null);
  return (
    <span className="flex items-center gap-1" title={`${list.length} window${list.length > 1 ? "s" : ""}`}>
      {list.map((v, i) => (
        <span key={i} className={`size-2 rounded-full ${v ? DOT[v] : "bg-line"}`} />
      ))}
    </span>
  );
}

export function SessionList({ items, maxHeight = 360 }: { items: Summary[]; maxHeight?: number }) {
  if (!items.length) return <p className="m-0 text-sm text-muted">None yet.</p>;
  return (
    <ul className="m-0 flex list-none flex-col gap-2 overflow-y-auto p-0 pr-1" style={{ maxHeight }}>
      {items.map((s) => (
        <li key={s.session_id}>
          <a
            href={`/session/${s.session_id}`}
            className={`flex items-center gap-4 rounded-xl border border-line border-l-4 bg-black/30 px-4 py-3 no-underline transition-colors hover:border-accent/50 hover:bg-black/50 ${EDGE[s.verdict] ?? "border-l-line"}`}
          >
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold text-ink">{firstSentence(s.reasoning)}</div>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-muted">
                <span>{when(s.created_at)}</span>
                <Dots verdicts={s.window_verdicts} count={s.windows} />
                {s.assurance && (
                  <span className={`rounded-full border px-1.5 py-px uppercase tracking-wider ${
                    s.assurance === "strong" ? "border-ok/50 text-ok" : "border-warn/50 text-warn"
                  }`}>
                    {s.assurance}
                  </span>
                )}
                {s.label && <span>test label {s.label}</span>}
              </div>
            </div>
            <span className={`shrink-0 font-mono text-base font-extrabold ${TEXT[s.verdict] ?? "text-muted"}`}>
              {s.verdict}
              <span className="ml-2 text-xs font-semibold text-muted">{score(s.score)}</span>
            </span>
          </a>
        </li>
      ))}
    </ul>
  );
}
