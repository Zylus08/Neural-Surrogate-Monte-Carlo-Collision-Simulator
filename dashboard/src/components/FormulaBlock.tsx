"use client";

import katex from "katex";
import { useRef, useEffect } from "react";

export default function FormulaBlock({
  math,
  label,
  className = "",
}: {
  math: string;
  label?: string;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current && math) {
      try {
        katex.render(math, containerRef.current, {
          displayMode: true,
          throwOnError: false,
          trust: true,
        });
      } catch {
        if (containerRef.current) {
          containerRef.current.textContent = math;
        }
      }
    }
  }, [math]);

  return (
    <div className={`glass-sm p-5 ${className}`}>
      <div ref={containerRef} className="overflow-x-auto" />
      {label && (
        <div className="text-xs text-gray-500 mt-2 font-medium tracking-wide">
          {label}
        </div>
      )}
    </div>
  );
}
