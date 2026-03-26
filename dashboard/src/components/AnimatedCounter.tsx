"use client";

import { useEffect, useRef } from "react";
import { useInView } from "framer-motion";

export default function AnimatedCounter({
  value,
  suffix = "",
  prefix = "",
  duration = 1.4,
  className = "",
}: {
  value: number;
  suffix?: string;
  prefix?: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!ref.current || !inView) return;
    const end = value;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = (now - startTime) / 1000;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      const current = Math.round(end * eased);

      let display: string;
      if (current >= 1e6) {
        display = `${(current / 1e6).toFixed(1)}M`;
      } else if (current >= 1e3) {
        display = current.toLocaleString();
      } else {
        display = current.toString();
      }

      ref.current!.textContent = `${prefix}${display}${suffix}`;
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [inView, value, duration, prefix, suffix]);

  return (
    <span ref={ref} className={className}>
      {prefix}0{suffix}
    </span>
  );
}
