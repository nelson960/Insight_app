import React, { useEffect, useMemo, useRef } from "react";

type Props = {
  left: React.ReactNode;
  right: React.ReactNode;
  ratio: number;
  onChangeRatio: (next: number) => void;
  minLeftPx?: number;
  minRightPx?: number;
  dividerPx?: number;
};

export function SplitView({
  left,
  right,
  ratio,
  onChangeRatio,
  minLeftPx = 260,
  minRightPx = 320,
  dividerPx = 10,
}: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const ratioRef = useRef(ratio);

  useEffect(() => {
    ratioRef.current = ratio;
  }, [ratio]);

  const r = useMemo(() => Math.max(0.05, Math.min(0.95, ratio)), [ratio]);

  function clampRatio(nextRatio: number, availableW: number) {
    const denom = Math.max(1, availableW);
    const minLeft = minLeftPx / denom;
    const minRight = minRightPx / denom;
    const max = 1 - minRight;
    const min = minLeft;
    return Math.max(min, Math.min(max, nextRatio));
  }

  function onDividerPointerDown(e: React.PointerEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const root = rootRef.current;
    if (!root) return;
    const rect = root.getBoundingClientRect();
    const startX = e.clientX;
    const startRatio = ratioRef.current;
    // Split proportions should be based on the remaining width (excluding the divider),
    // otherwise the panes + divider can exceed 100% and visually "clip" the right edge.
    const availableW = Math.max(1, rect.width - dividerPx);

    const pointerId = e.pointerId;

    const onMove = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      const dx = ev.clientX - startX;
      const next = startRatio + dx / availableW;
      onChangeRatio(clampRatio(next, availableW));
    };

    const onUp = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  }

  return (
    <div ref={rootRef} className="split-root">
      <div
        className="split-pane split-pane-left"
        style={{ flexGrow: r, flexShrink: 1, flexBasis: 0 }}
      >
        {left}
      </div>
      <div
        className="split-divider"
        style={{ width: dividerPx }}
        onPointerDown={onDividerPointerDown}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize panes"
      />
      <div
        className="split-pane split-pane-right"
        style={{ flexGrow: 1 - r, flexShrink: 1, flexBasis: 0 }}
      >
        {right}
      </div>
    </div>
  );
}
