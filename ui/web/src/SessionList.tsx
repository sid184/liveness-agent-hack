import { firstSentence, score, shape, VERDICT_COLOR, when, type Summary } from "./api";
import AnimatedList from "./reactbits/AnimatedList";

// Past calls (or windows) as a React Bits AnimatedList; selecting one opens its review page.
export function SessionList({ items, maxHeight = 360 }: { items: Summary[]; maxHeight?: number }) {
  if (!items.length) return <p className="m-0 text-sm text-muted">None yet.</p>;
  return (
    <AnimatedList
      items={items.map((s) => ({
        id: s.session_id,
        title: firstSentence(s.reasoning),
        description: `${when(s.created_at)} · ${shape(s)}${s.label ? ` · test label ${s.label}` : ""}`,
        meta: (
          <span className={`font-mono font-bold ${VERDICT_COLOR[s.verdict]}`}>
            {s.verdict} {score(s.score)}
          </span>
        ),
      }))}
      onItemSelect={(_, i) => (location.href = `/session/${items[i].session_id}`)}
      selectOnHover={false}
      frame={false}
      width="100%"
      maxHeight={maxHeight}
      accentColor="#5eead4"
    />
  );
}
