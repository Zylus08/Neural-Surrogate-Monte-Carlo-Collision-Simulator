"use client";

import { motion } from "framer-motion";
import {
  BookOpen,
  Atom,
  Zap,
  Orbit,
  ArrowRightLeft,
  Sigma,
  Dices,
} from "lucide-react";
import FormulaBlock from "@/components/FormulaBlock";
import StepTimeline from "@/components/StepTimeline";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.55,
      delay: i * 0.08,
      ease: [0.25, 0.4, 0.25, 1],
    },
  }),
};
const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

function Section({
  id,
  icon: Icon,
  number,
  color,
  title,
  children,
}: {
  id: string;
  icon: React.ElementType;
  number: string;
  color: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      id={id}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-80px" }}
      variants={stagger}
      className="scroll-mt-24"
    >
      <motion.div
        variants={fadeUp}
        custom={0}
        className="flex items-center gap-3 mb-6"
      >
        <span
          className={`flex items-center justify-center w-10 h-10 rounded-xl ${color} text-sm font-bold`}
          style={{
            backgroundColor: `color-mix(in srgb, currentColor 12%, transparent)`,
          }}
        >
          {number}
        </span>
        <Icon className={`w-5 h-5 ${color}`} />
        <h2 className="text-2xl md:text-3xl font-bold text-white">{title}</h2>
      </motion.div>
      {children}
    </motion.section>
  );
}

export default function TheoryPage() {
  const algorithmSteps = [
    {
      step: "1",
      title: "Sample beam energies",
      desc: "Draw E_A and E_B independently from a uniform distribution over the configured energy range.",
      formula:
        "E_A, E_B \\;\\sim\\; \\mathcal{U}(E_{\\min},\\, E_{\\max})",
      formulaLabel: "Uniform energy sampling",
    },
    {
      step: "2",
      title: "Sample collision angle (isotropic)",
      desc: "Use inverse-transform sampling on cos θ to respect the solid-angle element dΩ = sin θ dθ dφ.",
      formula:
        "\\cos\\theta \\sim \\mathcal{U}(\\cos\\theta_{\\max},\\, \\cos\\theta_{\\min}), \\qquad \\theta = \\arccos(\\cos\\theta)",
      formulaLabel: "Isotropic angular sampling",
    },
    {
      step: "3",
      title: "Compute 3-momenta",
      desc: "Use the energy–momentum relation and spherical-to-Cartesian conversion.",
      formula:
        "|\\mathbf{p}| = \\sqrt{E^2 - m^2}, \\qquad \\mathbf{p} = p\\,(\\sin\\theta\\cos\\phi,\\; \\sin\\theta\\sin\\phi,\\; \\cos\\theta)",
      formulaLabel: "Momentum calculation",
    },
    {
      step: "4",
      title: "Compute invariant mass",
      desc: "Evaluate √s from the total four-momentum of the system.",
      formula:
        "\\sqrt{s} = \\sqrt{(E_A + E_B)^2 - |\\mathbf{p}_A + \\mathbf{p}_B|^2}",
      formulaLabel: "Centre-of-mass energy",
    },
    {
      step: "5",
      title: "Final-state energies",
      desc: "Split the lab-frame total energy with a small stochastic asymmetry to model the random boost direction.",
      formula:
        "E_1 = (E_A + E_B)(0.5 + \\delta), \\quad E_2 = (E_A + E_B) - E_1, \\quad \\delta \\sim \\mathcal{U}(-0.1,\\, 0.1)",
      formulaLabel: "Energy splitting with asymmetry",
    },
    {
      step: "6",
      title: "Record event",
      desc: "Store (event_id, E_A, E_B, θ, p_x, p_y, p_z, E₁, E₂, √s) in the dataset.",
    },
  ];

  return (
    <div className="pt-28 pb-32 bg-grid">
      <div className="mx-auto max-w-4xl px-6 space-y-28">
        {/* Page header */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={stagger}
          className="text-center space-y-4"
        >
          <motion.span
            variants={fadeUp}
            custom={0}
            className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-accent-purple bg-accent-purple/10 border border-accent-purple/20 rounded-full px-4 py-1.5"
          >
            <BookOpen className="w-3.5 h-3.5" />
            Theoretical Foundations
          </motion.span>
          <motion.h1
            variants={fadeUp}
            custom={1}
            className="text-4xl sm:text-5xl font-extrabold tracking-tight section-heading"
          >
            Physics & Mathematics
          </motion.h1>
          <motion.p
            variants={fadeUp}
            custom={2}
            className="max-w-2xl mx-auto text-gray-400 leading-relaxed"
          >
            A guided walkthrough of every equation and physical concept that
            powers the simulator — from special relativity to Monte Carlo
            integration.
          </motion.p>
        </motion.div>

        {/* Monte Carlo Methods */}
        <Section
          id="monte-carlo"
          icon={Dices}
          number="01"
          color="text-accent-cyan"
          title="Monte Carlo Methods"
        >
          <motion.div
            variants={fadeUp}
            custom={1}
            className="space-y-4 text-gray-300 leading-relaxed"
          >
            <p>
              Monte Carlo methods are a class of computational algorithms that
              rely on{" "}
              <strong className="text-white">repeated random sampling</strong>{" "}
              to obtain numerical results. Named after the Monte Carlo Casino in
              Monaco, these techniques were formalised during the Manhattan
              Project by <em>Stanisław Ulam</em> and{" "}
              <em>John von Neumann</em> in the late 1940s.
            </p>
            <p>
              In high-energy physics, Monte Carlo simulations are
              indispensable. They are used to:
            </p>
            <ul className="list-none space-y-2 pl-4">
              {[
                "Generate pseudo-data that mimics detector output (event generators like Pythia, Herwig)",
                "Integrate high-dimensional phase-space integrals that are analytically intractable",
                "Propagate uncertainties by sampling nuisance parameters (systematic error estimation)",
              ].map((item, i) => (
                <motion.li
                  key={i}
                  variants={fadeUp}
                  custom={i + 2}
                  className="flex items-start gap-3"
                >
                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-accent-cyan shrink-0" />
                  <span>{item}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          <motion.div variants={fadeUp} custom={4} className="mt-6">
            <p className="text-sm text-gray-400 mb-2">
              The fundamental idea: estimate an expectation value by averaging
              over <em>N</em> random samples:
            </p>
            <FormulaBlock
              math="\langle f \rangle \;\approx\; \frac{1}{N} \sum_{i=1}^{N} f(x_i), \qquad x_i \sim p(x)"
              label="Monte Carlo estimator for ⟨f⟩ under distribution p(x)"
            />
          </motion.div>

          <motion.div variants={fadeUp} custom={5} className="mt-4">
            <p className="text-sm text-gray-400 mb-2">
              The statistical uncertainty decreases as 1/√N regardless of
              dimensionality — a key advantage over quadrature methods in
              high dimensions:
            </p>
            <FormulaBlock
              math="\sigma_{\langle f \rangle} = \frac{\sigma_f}{\sqrt{N}}"
              label="Standard error of the MC estimator"
            />
          </motion.div>
        </Section>

        {/* Special Relativity */}
        <Section
          id="relativity"
          icon={Zap}
          number="02"
          color="text-accent-blue"
          title="Special Relativity & Kinematics"
        >
          <motion.div
            variants={fadeUp}
            custom={1}
            className="space-y-4 text-gray-300 leading-relaxed"
          >
            <p>
              At the energies we simulate (GeV scale), particles travel at
              speeds close to <em>c</em>. Newtonian mechanics breaks down and we
              must use{" "}
              <strong className="text-white">
                Einstein&apos;s special relativity
              </strong>
              . We work in{" "}
              <strong className="text-white">natural units</strong> where ℏ = c
              = 1, so energy, momentum, and mass all have units of GeV.
            </p>
          </motion.div>

          <motion.div variants={fadeUp} custom={2}>
            <p className="text-sm text-gray-400 mb-2">
              The cornerstone is the{" "}
              <strong className="text-white">
                relativistic energy–momentum relation
              </strong>
              , which replaces E = ½mv²:
            </p>
            <FormulaBlock
              math="E^2 = |\mathbf{p}|^2 c^2 + m^2 c^4 \qquad \xrightarrow{c=1} \qquad E^2 = p^2 + m^2"
              label="Energy–momentum relation (natural units)"
            />
          </motion.div>

          <motion.div variants={fadeUp} custom={3}>
            <p className="text-sm text-gray-400 mb-2">
              From this we extract the 3-momentum magnitude:
            </p>
            <FormulaBlock
              math="|\mathbf{p}| = \sqrt{E^2 - m^2}"
              label="3-momentum magnitude"
            />
          </motion.div>

          <motion.div variants={fadeUp} custom={4}>
            <p className="text-sm text-gray-400 mb-2">
              We decompose the momentum into Cartesian components using
              spherical coordinates (θ = polar, φ = azimuthal):
            </p>
            <FormulaBlock
              math="p_x = p\sin\theta\cos\phi, \qquad p_y = p\sin\theta\sin\phi, \qquad p_z = p\cos\theta"
              label="Spherical → Cartesian decomposition"
            />
          </motion.div>

          <motion.div
            variants={fadeUp}
            custom={5}
            className="mt-4 glass-sm p-5"
          >
            <h4 className="text-sm font-semibold text-accent-blue mb-2">
              Why natural units?
            </h4>
            <p className="text-sm text-gray-400 leading-relaxed">
              Setting ℏ = c = 1 removes factors of 3×10⁸ from every equation.
              Energies, momenta, and masses all share the same unit (GeV),
              which simplifies both algebra and code. Every formula in this
              simulator uses natural units.
            </p>
          </motion.div>
        </Section>

        {/* Invariant Mass */}
        <Section
          id="invariant-mass"
          icon={Atom}
          number="03"
          color="text-accent-purple"
          title="Invariant Mass & Mandelstam s"
        >
          <motion.div
            variants={fadeUp}
            custom={1}
            className="space-y-4 text-gray-300 leading-relaxed"
          >
            <p>
              When two particles collide, the quantity that determines what new
              particles can be created is the{" "}
              <strong className="text-white">centre-of-mass energy</strong> √s.
              It is a <em>Lorentz invariant</em> — the same in every reference
              frame — and is defined through the{" "}
              <strong className="text-white">Mandelstam variable s</strong>:
            </p>
          </motion.div>

          <motion.div variants={fadeUp} custom={2}>
            <FormulaBlock
              math="s = \left(E_A + E_B\right)^2 - \left|\mathbf{p}_A + \mathbf{p}_B\right|^2"
              label="Mandelstam s — Lorentz-invariant squared CM energy"
            />
          </motion.div>

          <motion.div variants={fadeUp} custom={3}>
            <p className="text-sm text-gray-400 mb-2">
              Expanding the squared 3-momentum sum with the opening angle θ
              between the two beams:
            </p>
            <FormulaBlock
              math="s = m_A^2 + m_B^2 + 2\!\left(E_A E_B - |\mathbf{p}_A||\mathbf{p}_B|\cos\theta\right)"
              label="Expanded form with beam opening angle"
            />
          </motion.div>

          <motion.div
            variants={fadeUp}
            custom={4}
            className="mt-4 grid sm:grid-cols-2 gap-4"
          >
            <div className="glass-sm p-4">
              <h4 className="text-sm font-semibold text-green-400 mb-1">
                Head-on (θ = π)
              </h4>
              <p className="text-xs text-gray-400">
                cos θ = −1 → the cross-term is{" "}
                <strong className="text-white">maximised</strong>. This is the
                collider configuration (LHC, RHIC) that yields the highest √s
                for given beam energies.
              </p>
            </div>
            <div className="glass-sm p-4">
              <h4 className="text-sm font-semibold text-amber-400 mb-1">
                Fixed-target (θ ≈ 0)
              </h4>
              <p className="text-xs text-gray-400">
                cos θ ≈ 1 → the cross-term is{" "}
                <strong className="text-white">minimised</strong>. Much of the
                beam energy goes into boosting the CM system rather than
                creating particles.
              </p>
            </div>
          </motion.div>
        </Section>

        {/* CM Frame */}
        <Section
          id="cm-frame"
          icon={ArrowRightLeft}
          number="04"
          color="text-accent-pink"
          title="Centre-of-Mass Frame & Final States"
        >
          <motion.div
            variants={fadeUp}
            custom={1}
            className="space-y-4 text-gray-300 leading-relaxed"
          >
            <p>
              In the{" "}
              <strong className="text-white">
                centre-of-mass (CM) frame
              </strong>{" "}
              the total 3-momentum is zero. This makes the kinematics of the
              outgoing particles particularly simple. For a 2→2 process with
              outgoing masses m₁ and m₂, energy–momentum conservation uniquely
              determines the final-state energies:
            </p>
          </motion.div>

          <motion.div variants={fadeUp} custom={2}>
            <FormulaBlock
              math="E_1^* = \frac{s + m_1^2 - m_2^2}{2\sqrt{s}}, \qquad E_2^* = \sqrt{s} - E_1^*"
              label="Final-state energies in the CM frame"
            />
          </motion.div>

          <motion.div variants={fadeUp} custom={3}>
            <p className="text-sm text-gray-400 mb-2">
              When both outgoing particles are massless (e.g. γγ) this
              simplifies to an equal split:
            </p>
            <FormulaBlock
              math="E_1^* = E_2^* = \frac{\sqrt{s}}{2} \qquad (m_1 = m_2 = 0)"
              label="Ultra-relativistic limit"
            />
          </motion.div>

          <motion.div
            variants={fadeUp}
            custom={4}
            className="mt-4 glass-sm p-5"
          >
            <h4 className="text-sm font-semibold text-accent-pink mb-2">
              Lab-frame conservation in the simulator
            </h4>
            <p className="text-sm text-gray-400 leading-relaxed">
              Our simulator stores lab-frame quantities. To ensure exact energy
              conservation we split the <em>lab-frame</em> total energy E_A +
              E_B between the two outgoing particles with a stochastic
              asymmetry, rather than splitting √s. This guarantees E₁ + E₂ =
              E_A + E_B to machine precision (~10⁻¹⁷ GeV).
            </p>
          </motion.div>
        </Section>

        {/* Collider Geometry */}
        <Section
          id="geometry"
          icon={Orbit}
          number="05"
          color="text-green-400"
          title="Collider Geometry"
        >
          <motion.div
            variants={fadeUp}
            custom={1}
            className="space-y-4 text-gray-300 leading-relaxed"
          >
            <p>
              The simulator models a{" "}
              <strong className="text-white">symmetric collider</strong>:
              particle A travels along the +z axis and particle B along −z.
              The <em>crossing half-angle</em> θ parameterises how far each
              beam deviates from perfect head-on alignment:
            </p>
          </motion.div>

          <motion.div variants={fadeUp} custom={2}>
            <FormulaBlock
              math="\mathbf{p}_A = p_A\!\begin{pmatrix}\sin\theta\cos\phi_A \\ \sin\theta\sin\phi_A \\ \cos\theta\end{pmatrix}, \qquad \mathbf{p}_B = p_B\!\begin{pmatrix}\sin\theta\cos\phi_B \\ \sin\theta\sin\phi_B \\ -\cos\theta\end{pmatrix}"
              label="Counter-propagating beam momenta with crossing angle θ"
            />
          </motion.div>

          <motion.div variants={fadeUp} custom={3}>
            <p className="text-sm text-gray-400">
              The azimuthal angles φ_A and φ_B are drawn independently and
              uniformly in [0, 2π), breaking cylindrical symmetry when θ ≠ 0.
              For θ = 0 the beams are perfectly head-on and √s is maximised.
            </p>
          </motion.div>
        </Section>

        {/* Simulation Algorithm */}
        <Section
          id="algorithm"
          icon={Sigma}
          number="06"
          color="text-amber-400"
          title="Simulation Algorithm"
        >
          <motion.div
            variants={fadeUp}
            custom={1}
            className="text-gray-300 leading-relaxed mb-8"
          >
            <p>
              Putting everything together, here is the complete event-generation
              algorithm executed for each of the <em>N</em> collision events. In
              the implementation every step is{" "}
              <strong className="text-white">fully vectorised</strong> with
              NumPy, so all <em>N</em> events are processed simultaneously.
            </p>
          </motion.div>

          <StepTimeline steps={algorithmSteps} />
        </Section>
      </div>
    </div>
  );
}
