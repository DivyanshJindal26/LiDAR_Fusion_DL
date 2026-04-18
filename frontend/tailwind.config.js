/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg:       '#000000',
        surface:  '#0a0a0a',
        surface2: '#111111',
        accent:   '#00e676',
        accent2:  '#2979ff',
        accent3:  '#ff3d71',
        accent4:  '#ffab00',
        text:     '#f0f0f0',
        subtext:  '#555555',
        border:   'rgba(255,255,255,0.06)',
        brand: { DEFAULT: '#3b82f6', dim: '#1d4ed8' },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        slideInRight: {
          from: { transform: 'translateX(100%)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

