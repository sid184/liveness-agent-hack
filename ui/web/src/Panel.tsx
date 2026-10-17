import type { ReactNode } from "react";
import SpotlightCard from "./reactbits/SpotlightCard";

// Glass card over the animated backgrounds: SpotlightCard with a translucent, teal-edged surface.
export function Panel({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <SpotlightCard
      className={`p-6 backdrop-blur-md ${className}`}
      spotlightColor="rgba(94,234,212,.22)"
      borderGlow={0.9}
      style={{
        "--spotlight-card-surface": "rgba(10,14,18,.66)",
        "--spotlight-card-border":
          "linear-gradient(160deg, rgba(94,234,212,.38), rgba(94,234,212,.05) 45%, rgba(255,255,255,.07))",
        "--spotlight-card-shadow":
          "0 24px 60px -24px rgba(0,0,0,.9), inset 0 1px 0 rgba(255,255,255,.04)",
      } as React.CSSProperties}
    >
      {children}
    </SpotlightCard>
  );
}

export function PanelTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <h2 className="mt-0 mb-4 flex items-center justify-between gap-2 font-mono text-xs font-semibold uppercase tracking-widest text-muted">
      <span className="flex items-center gap-2">{children}</span>
      {aside}
    </h2>
  );
}
