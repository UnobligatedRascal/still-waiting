import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': 'http://192.168.137.29:9999',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
