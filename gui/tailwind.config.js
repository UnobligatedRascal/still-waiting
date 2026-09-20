/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'kepler': '#e8491d',
        'dark': '#0a0a0f',
        'panel': '#111118',
      },
    },
  },
  plugins: [],
}
