// LiveGuard mark: the G as an open ring with its crossbar, the overlay dot at its centre.
// The same ring sweeps while a check runs; the dot is the one that floats over the call.
export function GMark({
  size = 40,
  active = false,
  className = "",
}: {
  size?: number;
  active?: boolean;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label="LiveGuard"
    >
      <circle
        cx="50"
        cy="50"
        r="38"
        fill="none"
        stroke="currentColor"
        strokeWidth="11"
        strokeLinecap="round"
        strokeDasharray="196 43"
        className={active ? "lg-sweep" : ""}
      />
      <line
        x1="56"
        y1="50"
        x2="88"
        y2="50"
        stroke="currentColor"
        strokeWidth="11"
        strokeLinecap="round"
      />
      <circle cx="50" cy="50" r="7" fill="currentColor" className={active ? "lg-pulse" : ""} />
    </svg>
  );
}

export function Wordmark({ className = "" }: { className?: string }) {
  return (
    <span className={`font-mono font-extrabold tracking-tight ${className}`}>
      Live<span className="text-accent">G</span>uard
    </span>
  );
}
