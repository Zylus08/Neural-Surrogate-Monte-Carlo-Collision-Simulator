"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Atom, Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/theory", label: "Theory" },
  { href: "/simulation", label: "Simulation" },
  { href: "/results", label: "Results" },
];

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 w-full z-50 backdrop-blur-xl bg-black/40 border-b border-white/[0.06]">
      <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-3.5">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <motion.div
            whileHover={{ rotate: 180 }}
            transition={{ duration: 0.5 }}
            className="relative"
          >
            <Atom className="w-7 h-7 text-accent-cyan" />
            <div className="absolute inset-0 w-7 h-7 animate-ping-slow opacity-30">
              <Atom className="w-7 h-7 text-accent-purple" />
            </div>
          </motion.div>
          <span className="font-bold text-lg tracking-tight">
            <span className="text-accent-cyan">MC</span>{" "}
            <span className="text-gray-300">Simulator</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-1">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${active
                    ? "text-white"
                    : "text-gray-400 hover:text-white hover:bg-white/[0.04]"
                  }`}
              >
                {active && (
                  <motion.div
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg bg-white/[0.08] border border-white/[0.08]"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                  />
                )}
                <span className="relative z-10">{link.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Mobile toggle */}
        <button
          className="md:hidden text-gray-300 hover:text-white transition"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="md:hidden overflow-hidden border-t border-white/[0.06]"
          >
            <div className="px-6 py-4 space-y-1 bg-black/60 backdrop-blur-xl">
              {links.map((link) => {
                const active = pathname === link.href;
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileOpen(false)}
                    className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition ${active
                        ? "text-white bg-white/[0.08]"
                        : "text-gray-400 hover:text-white hover:bg-white/[0.04]"
                      }`}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
