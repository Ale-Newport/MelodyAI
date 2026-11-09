import { defineConfig } from "vitest/config";
import type { Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

/**
 * `--mode pages` builds the static demo for GitHub Pages (https://ale-newport.github.io/MelodyAI/):
 * served under /MelodyAI/, with no API server (see src/lib/demo.ts). Every other build leaves the
 * captured demo files (public/demo) out of dist.
 */
function demoFilesOnlyInPages(pages: boolean): Plugin {
  return {
    name: "melodyai-demo-files",
    apply: "build",
    closeBundle() {
      if (!pages) fs.rmSync(path.resolve(__dirname, "dist", "demo"), { recursive: true, force: true });
    },
  };
}

export default defineConfig(({ mode }) => {
  const pages = mode === "pages";
  return {
    base: pages ? "/MelodyAI/" : "/",
    plugins: [react(), demoFilesOnlyInPages(pages)],
    define: pages ? { "import.meta.env.VITE_DEMO": JSON.stringify("1") } : {},
    resolve: { alias: { "@": path.resolve(__dirname, "src") } },
    server: {
      port: 5173,
      proxy: { "/api": { target: process.env.MELODYAI_API ?? "http://127.0.0.1:8000", changeOrigin: true } },
    },
    build: { outDir: "dist", chunkSizeWarningLimit: 1200 },
    test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"], globals: true, css: false },
  };
});
