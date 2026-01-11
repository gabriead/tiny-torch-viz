---
layout: home

hero:
  name: "TinyTorch Viz"
  text: "Deep Learning, Visualized."
  tagline: Don't just run code. Watch the matrices learn.
  actions:
    - theme: brand
      text: Launch App
      link: /app/index.html
      target: _blank
    - theme: alt
      text: Read the Guide
      link: /guide/what-is-it

features:
  - title: Live Execution
    details: Write NumPy or TinyTorch code and see the execution graph build instantly.
  - title: Interactive Grids
    details: Don't just print tensors. Watch row-column multiplications happen step-by-step.
  - title: Education First
    details: Built for learners to bridge the gap between "Math" and "Code".
---

<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
  {
    avatar: 'https://github.com/gabriead.png',
    name: 'Gabriel Adrian',
    title: 'Creator',
    links: [
      { icon: 'github', link: 'https://github.com/gabriead' }
    ]
  }
]
</script>

## Inspired by the Community

This project bridges the gap between [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html) and [AI by Hand](https://www.byhand.ai/).

<div style="margin-top: 2rem;">
  <VPTeamMembers size="small" :members="members" />
</div>