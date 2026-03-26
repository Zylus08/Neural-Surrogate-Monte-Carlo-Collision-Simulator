import type { Metadata } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "Monte Carlo Particle Collision Simulator",
  description:
    "Interactive dashboard for a production-quality Monte Carlo simulator of high-energy particle collisions. Explore physics theory, run simulations, and visualize results.",
  keywords: ["Monte Carlo", "particle physics", "simulation", "collision", "dashboard"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        <Navbar />
        <main className="relative z-10 min-h-screen">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
