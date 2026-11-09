/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono Variable"', "ui-monospace", "monospace"],
      },
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        elevated: "rgb(var(--elevated) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        brand: { DEFAULT: "rgb(var(--brand) / <alpha-value>)", fg: "rgb(var(--brand-fg) / <alpha-value>)" },
        highlight: "rgb(var(--highlight) / <alpha-value>)",
      },
      borderRadius: { lg: "6px", xl: "8px", "2xl": "12px", "3xl": "16px" },
      keyframes: {
        ripple: { "0%": { transform: "scale(1)", opacity: "0.5" }, "100%": { transform: "scale(1.9)", opacity: "0" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        indeterminate: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(250%)" } },
      },
      animation: {
        ripple: "ripple 2.4s cubic-bezier(0.2,0.6,0.3,1) infinite",
        indeterminate: "indeterminate 1.4s ease-in-out infinite",
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
