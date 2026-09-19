import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

const token = (name: string) => `rgb(var(--m-${name}) / <alpha-value>)`;

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["Space Grotesk", "Inter", "ui-sans-serif", "sans-serif"],
      },
      spacing: {
        "4.5": "1.125rem",
      },
      colors: {
        paper: token("paper"),
        surface: token("surface"),
        raised: token("raised"),
        line: token("line"),
        ink: {
          DEFAULT: token("ink"),
          dim: token("ink-dim"),
          muted: token("ink-muted"),
        },
        blue: token("blue"),
        green: token("green"),
        red: token("red"),
      },
      keyframes: {
        "trace-ping": {
          "0%": { transform: "scale(1)", opacity: ".5" },
          "80%, 100%": { transform: "scale(1.9)", opacity: "0" },
        },
        scan: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100vw)" },
        },
      },
      animation: {
        "trace-ping": "trace-ping 1.4s cubic-bezier(0, 0, 0.2, 1) infinite",
        scan: "scan 2.4s cubic-bezier(0.4, 0, 0.2, 1) infinite",
      },
    },
  },
  plugins: [animate],
} satisfies Config;
