"use client";

import { motion } from "framer-motion";
import FormulaBlock from "./FormulaBlock";

interface Step {
  step: string;
  title: string;
  desc: string;
  formula?: string;
  formulaLabel?: string;
}

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, delay: i * 0.1, ease: [0.25, 0.4, 0.25, 1] },
  }),
};

export default function StepTimeline({ steps }: { steps: Step[] }) {
  return (
    <motion.div
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      className="relative"
    >
      {steps.map((s, i) => (
        <motion.div
          key={s.step}
          variants={fadeUp}
          custom={i}
          className="flex gap-5 mb-8 last:mb-0"
        >
          {/* Timeline column */}
          <div className="flex flex-col items-center shrink-0">
            <div
              className="timeline-dot border border-amber-400/30 text-amber-400"
              style={{ background: "rgba(245, 158, 11, 0.08)" }}
            >
              {s.step}
            </div>
            {i < steps.length - 1 && (
              <div className="flex-1 w-px mt-2 timeline-line" />
            )}
          </div>

          {/* Content */}
          <div className="pb-4 flex-1 min-w-0">
            <h4 className="text-base font-semibold text-white mb-1.5">
              {s.title}
            </h4>
            <p className="text-sm text-gray-400 mb-3 leading-relaxed">
              {s.desc}
            </p>
            {s.formula && (
              <FormulaBlock
                math={s.formula}
                label={s.formulaLabel}
                className="text-sm"
              />
            )}
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}
