"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

export default function GlowCard({
  children,
  className = "",
  glowColor = "cyan",
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  glowColor?: "cyan" | "purple" | "blue" | "pink" | "green" | "amber";
  delay?: number;
}) {
  const glowMap: Record<string, string> = {
    cyan: "stat-glow-cyan",
    purple: "stat-glow-purple",
    blue: "stat-glow-blue",
    pink: "stat-glow-pink",
    green: "stat-glow-green",
    amber: "stat-glow-amber",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.97 }}
      whileInView={{ opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{
        duration: 0.5,
        delay,
        ease: [0.25, 0.4, 0.25, 1],
      }}
      whileHover={{ scale: 1.02, y: -4 }}
      className={`glass-sm glow-card p-6 ${glowMap[glowColor]} ${className}`}
    >
      {children}
    </motion.div>
  );
}
