import { defineConfig } from 'vitepress'

export default defineConfig({
  // IMPORTANT: Replace 'aibyhand' with your REPO name if it changes!
  base: '/tiny-torch-viz/',

  title: "TinyTorch Viz",
  description: "Interactive Deep Learning Visualization",

  themeConfig: {
    // Top Navigation Bar
    nav: [
      { text: 'Guide', link: '/guide/what-is-it' },
      { text: 'Launch App', link: '/app/index.html', target: '_blank' },
      { text: 'GitHub', link: 'https://github.com/gabriead/aibyhand' }
    ],

    // Side Navigation Bar
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/guide/what-is-it' },
          { text: 'Quick Start', link: '/guide/what-is-it#quick-start' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Tensors', link: '/guide/what-is-it#tensor' },
          { text: 'Layers', link: '/guide/what-is-it#layers' },
          { text: 'Optimizers', link: '/guide/what-is-it#optimizers' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/gabriead/tiny-torch-viz' }
    ]
  }
})
