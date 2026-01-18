import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;
// Allow port override via environment variable (defaults to 1420)
const port = Number(process.env.TAURI_DEV_PORT) || 1420;
// Allow strict port to be disabled if needed (useful when port is in use)
const strictPort = process.env.TAURI_STRICT_PORT !== "false";

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, but can fall back to alternative port if configured
  server: {
    port,
    strictPort,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: port + 1,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
  // 4. Build directly into src-tauri so Tauri can find the files when bundled
  build: {
    outDir: "./src-tauri/dist",
    emptyOutDir: true,
  },
}));
