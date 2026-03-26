import { Atom, Github, BookOpen, Zap } from "lucide-react";
import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full border-t border-white/[0.06] bg-black/30 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Atom className="w-5 h-5 text-accent-cyan" />
              <span className="font-bold text-white">MC Simulator</span>
            </div>
            <p className="text-sm text-gray-500 leading-relaxed">
              A production-quality Monte Carlo simulator for simplified high-energy particle collisions.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-3">Explore</h4>
            <div className="space-y-2">
              <Link href="/theory" className="flex items-center gap-2 text-sm text-gray-500 hover:text-accent-purple transition">
                <BookOpen className="w-3.5 h-3.5" /> Theory & Formulas
              </Link>
              <Link href="/simulation" className="flex items-center gap-2 text-sm text-gray-500 hover:text-accent-cyan transition">
                <Zap className="w-3.5 h-3.5" /> Run Simulation
              </Link>
              <Link href="/results" className="flex items-center gap-2 text-sm text-gray-500 hover:text-accent-green transition">
                <Github className="w-3.5 h-3.5" /> View Results
              </Link>
            </div>
          </div>

          {/* Tech stack */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-3">Tech Stack</h4>
            <div className="flex flex-wrap gap-2">
              {[
                { label: "Next.js", color: "text-white bg-white/[0.06]" },
                { label: "TypeScript", color: "text-blue-400 bg-blue-500/10" },
                { label: "Python", color: "text-yellow-400 bg-yellow-500/10" },
                { label: "NumPy", color: "text-cyan-400 bg-cyan-500/10" },
                { label: "SciPy", color: "text-orange-400 bg-orange-500/10" },
                { label: "Framer Motion", color: "text-purple-400 bg-purple-500/10" },
              ].map((tag) => (
                <span
                  key={tag.label}
                  className={`text-xs font-medium px-2.5 py-1 rounded-md ${tag.color}`}
                >
                  {tag.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="section-divider mb-6" />
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
          <span className="text-xs text-gray-600">
            © 2026 Monte Carlo Particle Collision Simulator
          </span>
          <span className="text-xs text-gray-600">
            Built with science & code
          </span>
        </div>
      </div>
    </footer>
  );
}
