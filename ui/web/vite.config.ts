// Builds into ../static, which FastAPI serves at /static. On Vercel (VERCEL=1 during its build)
// it builds into dist at the site root instead. `npm run dev` proxies the API to
// a running `make ui` so the pages can be edited with hot reload.
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { defineConfig } from "vite";

const api = "http://localhost:8765";
const vercel = !!process.env.VERCEL;
if (vercel) process.env.VITE_API_BASE ??= api;

export default defineConfig({
  base: vercel ? "/" : "/static/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: vercel ? "dist" : "../static",
    emptyOutDir: true,
    rollupOptions: {
      input: { index: resolve(import.meta.dirname, "index.html"), session: resolve(import.meta.dirname, "session.html") },
    },
  },
  server: { proxy: { "/api": api, "/media": api } },
});
