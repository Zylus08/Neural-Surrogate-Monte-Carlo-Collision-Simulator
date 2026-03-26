"use client";

import { motion } from "framer-motion";
import {
  Atom,
  Zap,
  BarChart3,
  ArrowRight,
  BookOpen,
  Play,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import ParticleCollisionCanvas from "@/components/ParticleCollisionCanvas";
import AnimatedCounter from "@/components/AnimatedCounter";
import GlowCard from "@/components/GlowCard";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay: i * 0.1, ease: [0.25, 0.4, 0.25, 1] },
  }),
};

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};

export default function HomePage() {
  return (
    <div className="relative">
      <ParticleCollisionCanvas />

      {/* Hero Section */}
      <section className="relative z-10 min-h-screen flex items-center justify-center">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <motion.div
            initial="hidden"
            animate="show"
            variants={stagger}
            className="space-y-6"
          >
            {/* Badge */}
            <motion.div variants={fadeUp} custom={0}>
              <span className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-accent-cyan bg-accent-cyan/10 border border-accent-cyan/20 rounded-full px-4 py-1.5 font-medium">
                <Sparkles className="w-3.5 h-3.5" />
                High-Energy Physics Simulation
              </span>
            </motion.div>

            {/* Title */}
            <motion.h1
              variants={fadeUp}
              custom={1}
              className="text-5xl sm:text-6xl md:text-7xl font-black tracking-tight leading-[1.1]"
            >
              <span className="section-heading">Monte Carlo</span>
              <br />
              <span className="text-white">Particle Collision</span>
              <br />
              <span className="text-gray-400 text-4xl sm:text-5xl md:text-6xl font-bold">
                Simulator
              </span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              variants={fadeUp}
              custom={2}
              className="max-w-2xl mx-auto text-lg text-gray-400 leading-relaxed"
            >
              Generate 100,000+ collision events in seconds. Explore relativistic
              kinematics, conservation laws, and statistical analysis — all
              powered by Monte Carlo methods.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              variants={fadeUp}
              custom={3}
              className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4"
            >
              <Link
                href="/simulation"
                className="btn-primary px-8 py-3.5 rounded-xl text-base flex items-center gap-2"
              >
                <span className="flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Run Simulation
                </span>
              </Link>
              <Link
                href="/theory"
                className="btn-secondary px-8 py-3.5 text-base flex items-center gap-2"
              >
                <BookOpen className="w-4 h-4" />
                Explore Theory
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 py-24 bg-grid">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6"
          >
            {[
              {
                label: "Events Generated",
                value: 100000,
                color: "cyan" as const,
                textClass: "text-accent-cyan",
              },
              {
                label: "Throughput",
                value: 670000,
                suffix: "/s",
                color: "blue" as const,
                textClass: "text-accent-blue",
              },
              {
                label: "Conservation",
                display: "10⁻¹⁷ GeV",
                color: "purple" as const,
                textClass: "text-accent-purple",
              },
              {
                label: "Plot Types",
                display: "6",
                color: "green" as const,
                textClass: "text-accent-green",
              },
            ].map((stat, i) => (
              <GlowCard key={stat.label} glowColor={stat.color} delay={i * 0.1}>
                <div className="text-center">
                  <div className="text-xs text-gray-500 font-medium tracking-wide uppercase mb-2">
                    {stat.label}
                  </div>
                  {"value" in stat && stat.value ? (
                    <AnimatedCounter
                      value={stat.value}
                      suffix={stat.suffix}
                      className={`text-3xl font-bold ${stat.textClass}`}
                    />
                  ) : (
                    <span className={`text-3xl font-bold ${stat.textClass}`}>
                      {stat.display}
                    </span>
                  )}
                </div>
              </GlowCard>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Pipeline Section */}
      <section className="relative z-10 py-24">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
            className="text-center mb-16"
          >
            <motion.h2
              variants={fadeUp}
              custom={0}
              className="text-3xl md:text-4xl font-bold text-white mb-4"
            >
              Simulation <span className="section-heading">Pipeline</span>
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={1}
              className="text-gray-400 max-w-xl mx-auto"
            >
              From random sampling to publication-quality visualizations in four
              stages.
            </motion.p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                icon: Atom,
                title: "Monte Carlo Sampling",
                desc: "Draw beam energies from uniform distributions and collision angles from isotropic distributions.",
                color: "text-accent-cyan",
                borderColor: "border-accent-cyan/20",
              },
              {
                icon: Zap,
                title: "Relativistic Kinematics",
                desc: "Calculate 3-momenta, invariant mass √s, and final-state energies using conservation laws.",
                color: "text-accent-purple",
                borderColor: "border-accent-purple/20",
              },
              {
                icon: BarChart3,
                title: "Dataset Generation",
                desc: "Package event data into structured arrays with CSV/NumPy export capabilities.",
                color: "text-accent-blue",
                borderColor: "border-accent-blue/20",
              },
              {
                icon: Sparkles,
                title: "Analysis & Plots",
                desc: "Distribution fitting, KS goodness-of-fit tests, and 6 publication-quality figures.",
                color: "text-accent-green",
                borderColor: "border-accent-green/20",
              },
            ].map((stage, i) => (
              <GlowCard
                key={stage.title}
                glowColor={
                  (["cyan", "purple", "blue", "green"] as const)[i]
                }
                delay={i * 0.12}
                className={`border-l-2 ${stage.borderColor}`}
              >
                <div className="flex items-start gap-3 mb-3">
                  <span
                    className={`flex items-center justify-center w-8 h-8 rounded-lg ${stage.color} bg-current/10 shrink-0`}
                    style={{ backgroundColor: `color-mix(in srgb, currentColor 10%, transparent)` }}
                  >
                    <stage.icon className="w-4 h-4" />
                  </span>
                  <span className="text-xs text-gray-600 font-mono mt-1">
                    0{i + 1}
                  </span>
                </div>
                <h3 className={`font-semibold text-white mb-2`}>
                  {stage.title}
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  {stage.desc}
                </p>
              </GlowCard>
            ))}
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="relative z-10 py-24 bg-grid">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={stagger}
            className="text-center mb-16"
          >
            <motion.h2
              variants={fadeUp}
              custom={0}
              className="text-3xl md:text-4xl font-bold text-white mb-4"
            >
              Explore the <span className="section-heading">Physics</span>
            </motion.h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                title: "Theory & Formulas",
                desc: "Dive into special relativity, Mandelstam variables, CM frame kinematics, and the complete simulation algorithm.",
                href: "/theory",
                icon: BookOpen,
                color: "purple" as const,
              },
              {
                title: "Interactive Simulation",
                desc: "Configure parameters, run thousands of collision events, and see real-time statistical results.",
                href: "/simulation",
                icon: Zap,
                color: "cyan" as const,
              },
              {
                title: "Visualizations",
                desc: "Energy distributions, angular plots, invariant mass spectra, and energy correlation charts.",
                href: "/results",
                icon: BarChart3,
                color: "green" as const,
              },
            ].map((card, i) => (
              <Link key={card.title} href={card.href}>
                <GlowCard glowColor={card.color} delay={i * 0.1} className="h-full">
                  <card.icon
                    className={`w-8 h-8 mb-4 text-accent-${card.color}`}
                  />
                  <h3 className="text-lg font-bold text-white mb-2">
                    {card.title}
                  </h3>
                  <p className="text-sm text-gray-400 leading-relaxed mb-4">
                    {card.desc}
                  </p>
                  <span
                    className={`inline-flex items-center gap-1 text-sm font-medium text-accent-${card.color}`}
                  >
                    Explore <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </GlowCard>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
