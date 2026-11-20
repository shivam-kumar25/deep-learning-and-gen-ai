import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = `http://localhost:${process.env.PORT || 4000}`;

export default defineConfig({
  root: "client",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": API, "/content": API },
  },
  build: { outDir: "../dist", emptyOutDir: true },
});
