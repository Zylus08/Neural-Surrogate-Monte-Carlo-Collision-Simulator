"use client";

import { useEffect, useRef, useCallback } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  trail: { x: number; y: number; alpha: number }[];
  life: number;
  maxLife: number;
  type: "beam-a" | "beam-b" | "collision" | "ambient";
}

export default function ParticleCollisionCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  const mouse = useRef({ x: -999, y: -999 });
  const particles = useRef<Particle[]>([]);
  const collisionFlashes = useRef<{ x: number; y: number; radius: number; alpha: number; color: string }[]>([]);
  const frameRef = useRef(0);

  const createBeamParticle = useCallback((w: number, h: number, type: "beam-a" | "beam-b"): Particle => {
    const cx = w / 2;
    const cy = h / 2;
    const isA = type === "beam-a";
    const startX = isA ? -20 : w + 20;
    const startY = cy + (Math.random() - 0.5) * h * 0.3;
    const speed = 1.5 + Math.random() * 2;
    const colors = isA
      ? ["#00eaff", "#00d4ff", "#00bfff", "#38bdf8"]
      : ["#a855f7", "#c084fc", "#d946ef", "#ec4899"];

    return {
      x: startX,
      y: startY,
      vx: isA ? speed : -speed,
      vy: (Math.random() - 0.5) * 0.5,
      radius: 1.5 + Math.random() * 2,
      color: colors[Math.floor(Math.random() * colors.length)],
      trail: [],
      life: 0,
      maxLife: 300 + Math.random() * 200,
      type,
    };
  }, []);

  const createCollisionParticle = useCallback((x: number, y: number): Particle => {
    const angle = Math.random() * Math.PI * 2;
    const speed = 1 + Math.random() * 4;
    const colors = ["#00eaff", "#a855f7", "#ec4899", "#f59e0b", "#22c55e", "#ffffff"];
    return {
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      radius: 1 + Math.random() * 2.5,
      color: colors[Math.floor(Math.random() * colors.length)],
      trail: [],
      life: 0,
      maxLife: 40 + Math.random() * 60,
      type: "collision",
    };
  }, []);

  const createAmbientParticle = useCallback((w: number, h: number): Particle => {
    const colors = ["#00eaff", "#a855f7", "#3b82f6", "#ec4899"];
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      radius: 1 + Math.random() * 1.5,
      color: colors[Math.floor(Math.random() * colors.length)],
      trail: [],
      life: 0,
      maxLife: Infinity,
      type: "ambient",
    };
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const DPR = window.devicePixelRatio || 1;
    let width = window.innerWidth;
    let height = window.innerHeight;

    function resize() {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas!.width = width * DPR;
      canvas!.height = height * DPR;
      canvas!.style.width = width + "px";
      canvas!.style.height = height + "px";
      ctx!.scale(DPR, DPR);
    }
    resize();

    // Init ambient particles
    for (let i = 0; i < 40; i++) {
      particles.current.push(createAmbientParticle(width, height));
    }

    function spawnBeams() {
      if (Math.random() < 0.15) {
        particles.current.push(createBeamParticle(width, height, "beam-a"));
      }
      if (Math.random() < 0.15) {
        particles.current.push(createBeamParticle(width, height, "beam-b"));
      }
    }

    function checkCollisions() {
      const cx = width / 2;
      const collisionZone = 60;
      const beamsA = particles.current.filter(p => p.type === "beam-a" && Math.abs(p.x - cx) < collisionZone);
      const beamsB = particles.current.filter(p => p.type === "beam-b" && Math.abs(p.x - cx) < collisionZone);

      for (const a of beamsA) {
        for (const b of beamsB) {
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 15) {
            // Create collision burst
            const mx = (a.x + b.x) / 2;
            const my = (a.y + b.y) / 2;
            for (let k = 0; k < 8 + Math.floor(Math.random() * 8); k++) {
              particles.current.push(createCollisionParticle(mx, my));
            }
            collisionFlashes.current.push({
              x: mx, y: my, radius: 3, alpha: 1,
              color: Math.random() > 0.5 ? "#00eaff" : "#a855f7",
            });
            a.life = a.maxLife;
            b.life = b.maxLife;
            break;
          }
        }
      }
    }

    function update() {
      spawnBeams();
      checkCollisions();

      const mx = mouse.current.x;
      const my = mouse.current.y;

      for (const p of particles.current) {
        // Trail
        if (p.type !== "ambient") {
          p.trail.unshift({ x: p.x, y: p.y, alpha: 1 });
          if (p.trail.length > 12) p.trail.pop();
          for (const t of p.trail) t.alpha *= 0.85;
        }

        // Mouse interaction for ambient
        if (p.type === "ambient" && mx > 0 && my > 0) {
          const dx = p.x - mx;
          const dy = p.y - my;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120 && dist > 0) {
            const force = 0.5 / dist;
            p.vx += dx * force;
            p.vy += dy * force;
          }
        }

        // Damping for ambient
        if (p.type === "ambient") {
          p.vx *= 0.99;
          p.vy *= 0.99;
        }

        // Collision particles slow down
        if (p.type === "collision") {
          p.vx *= 0.97;
          p.vy *= 0.97;
        }

        p.x += p.vx;
        p.y += p.vy;
        p.life++;

        // Wrap ambient
        if (p.type === "ambient") {
          if (p.x < -10) p.x = width + 10;
          if (p.x > width + 10) p.x = -10;
          if (p.y < -10) p.y = height + 10;
          if (p.y > height + 10) p.y = -10;
        }
      }

      // Remove dead particles
      particles.current = particles.current.filter(p => p.life < p.maxLife);

      // Update flashes
      for (const f of collisionFlashes.current) {
        f.radius += 3;
        f.alpha *= 0.9;
      }
      collisionFlashes.current = collisionFlashes.current.filter(f => f.alpha > 0.01);
    }

    function draw() {
      ctx!.clearRect(0, 0, width, height);

      // Draw connecting lines for ambient particles
      const ambients = particles.current.filter(p => p.type === "ambient");
      for (let i = 0; i < ambients.length; i++) {
        for (let j = i + 1; j < ambients.length; j++) {
          const dx = ambients[i].x - ambients[j].x;
          const dy = ambients[i].y - ambients[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            ctx!.save();
            ctx!.globalAlpha = 0.06 * (1 - dist / 150);
            ctx!.beginPath();
            ctx!.moveTo(ambients[i].x, ambients[i].y);
            ctx!.lineTo(ambients[j].x, ambients[j].y);
            ctx!.strokeStyle = "#a855f7";
            ctx!.lineWidth = 0.5;
            ctx!.stroke();
            ctx!.restore();
          }
        }
      }

      // Draw collision flashes
      for (const f of collisionFlashes.current) {
        ctx!.save();
        ctx!.globalAlpha = f.alpha;
        const grad = ctx!.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.radius);
        grad.addColorStop(0, f.color);
        grad.addColorStop(0.5, f.color + "40");
        grad.addColorStop(1, "transparent");
        ctx!.fillStyle = grad;
        ctx!.beginPath();
        ctx!.arc(f.x, f.y, f.radius, 0, Math.PI * 2);
        ctx!.fill();
        ctx!.restore();
      }

      // Draw particles
      for (const p of particles.current) {
        const lifeRatio = p.type !== "ambient" ? 1 - p.life / p.maxLife : 1;

        // Draw trail
        for (let i = 0; i < p.trail.length; i++) {
          const t = p.trail[i];
          ctx!.save();
          ctx!.globalAlpha = t.alpha * lifeRatio * 0.4;
          ctx!.beginPath();
          ctx!.arc(t.x, t.y, p.radius * (1 - i / p.trail.length) * 0.7, 0, Math.PI * 2);
          ctx!.fillStyle = p.color;
          ctx!.fill();
          ctx!.restore();
        }

        // Draw particle
        ctx!.save();
        ctx!.globalAlpha = lifeRatio * (p.type === "ambient" ? 0.6 : 0.9);
        ctx!.beginPath();
        ctx!.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx!.fillStyle = p.color;
        ctx!.shadowColor = p.color;
        ctx!.shadowBlur = p.type === "collision" ? 15 : 8;
        ctx!.fill();
        ctx!.restore();
      }

      // Draw center collision zone glow
      const cx = width / 2;
      const cy = height / 2;
      const pulse = 0.3 + 0.1 * Math.sin(frameRef.current * 0.02);
      ctx!.save();
      ctx!.globalAlpha = pulse;
      const zoneGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, 80);
      zoneGrad.addColorStop(0, "rgba(168, 85, 247, 0.15)");
      zoneGrad.addColorStop(0.5, "rgba(0, 234, 255, 0.05)");
      zoneGrad.addColorStop(1, "transparent");
      ctx!.fillStyle = zoneGrad;
      ctx!.beginPath();
      ctx!.arc(cx, cy, 80, 0, Math.PI * 2);
      ctx!.fill();
      ctx!.restore();
    }

    function loop() {
      frameRef.current++;
      update();
      draw();
      requestAnimationFrame(loop);
    }

    const animId = requestAnimationFrame(loop);

    function handleMouse(e: MouseEvent) {
      mouse.current = { x: e.clientX, y: e.clientY };
    }

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", handleMouse);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouse);
    };
  }, [createAmbientParticle, createBeamParticle, createCollisionParticle]);

  return (
    <canvas
      ref={ref}
      className="fixed inset-0 z-0 pointer-events-none"
      style={{ opacity: 0.7 }}
      aria-hidden="true"
    />
  );
}
