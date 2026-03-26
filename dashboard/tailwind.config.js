/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: {
          cyan: "#00eaff",
          blue: "#3b82f6",
          purple: "#a855f7",
          pink: "#ec4899",
          green: "#22c55e",
          amber: "#f59e0b",
        },
        surface: {
          DEFAULT: "#0a0a1a",
          light: "#12122a",
          lighter: "#1a1a3a",
        }
      },
      animation: {
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'fade-up': 'fadeUp 0.7s cubic-bezier(0.25,0.4,0.25,1)',
        'orbit': 'orbit 6s linear infinite',
        'beam': 'beamAnim 1.5s ease-in-out infinite',
        'spin-slow': 'spin 8s linear infinite',
        'ping-slow': 'ping 3s cubic-bezier(0, 0, 0.2, 1) infinite',
        'gradient-x': 'gradientX 4s ease infinite',
        'particle-float': 'particleFloat 6s ease-in-out infinite',
      },
      keyframes: {
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(0, 234, 255, 0.15)' },
          '50%': { boxShadow: '0 0 40px rgba(168, 85, 247, 0.3)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg) translateX(12px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(12px) rotate(-360deg)' },
        },
        beamAnim: {
          '0%, 100%': { opacity: '0.3', transform: 'scaleX(0.5)' },
          '50%': { opacity: '1', transform: 'scaleX(1)' },
        },
        gradientX: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        particleFloat: {
          '0%, 100%': { transform: 'translate(0, 0) rotate(0deg)' },
          '25%': { transform: 'translate(10px, -15px) rotate(90deg)' },
          '50%': { transform: 'translate(-5px, -25px) rotate(180deg)' },
          '75%': { transform: 'translate(-15px, -10px) rotate(270deg)' },
        },
      },
      backgroundSize: {
        'shimmer': '200% 100%',
      },
    },
  },
  plugins: [],
};
