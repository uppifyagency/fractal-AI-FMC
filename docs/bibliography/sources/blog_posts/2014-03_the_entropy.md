---
title: "The Entropy"
author: Sergio Hernández Cerezo
date: 2014-03
url: http://entropicai.blogspot.com/2014/03/the-enthropy.html
fetched: 2026-04-26
---

# The Entropy

## Introduction

The blog describes how intelligent behavior can be defined using thermodynamic principles centered on entropy. Complex formulas underpin the concept, yet implementation proves simpler than expected. Fractal approaches further enhance the system by reducing computational demands.

## Main Content

### Definition of Entropy

The post employs an accessible analogy: imagine a transparent sphere containing 100 bees. The visible state constitutes the "macro state," while individual bee positions at frozen moments represent "micro states." Entropy (S = k Log(N)) measures the quantity of possible distinct images or configurations.

Reducing the sphere's radius decreases possible configurations. Maximum order (minimum entropy) occurs when movement becomes impossible—only one image remains possible.

### Connection to Physical Laws

The second law of thermodynamics states that isolated systems exhibit increasing entropy over time. More precisely, systems evolve by maximizing entropy growth at each moment.

> "a system always evolves in exactly the way its entropy grows as much as possible"

This principle explains classical physics phenomena. Note: this applies to macroscopic systems with numerous particles, not quantum nano-systems where entropy occasionally decreases.

### Life and Entropy

Living organisms lower their internal entropy while increasing environmental entropy. A definition offered:

> "When a subsystem capable of consistently lowering the subsystem entropy...is said that this subsystem is alive"

### Intelligence and Entropy

Intelligent behavior emerges from maximizing accessible future states. Decision-making between options involves counting reachable futures per choice—the option yielding more futures represents the superior decision.

This "entropy of futures" concept—technically termed "causal entropic force"—guides intelligent agents toward preserving possibilities and flexibility.

### Implementation Note

The algorithm represents a simplified approach rather than strict entropy calculation, yet it demonstrates remarkable versatility across diverse applications.

---

## Annotation (analisi storica)

**Importanza nel corpus**: post fondazionale. Espone in modo informale ma rigoroso le tre idee-cardine che diventeranno il paper 2018:

1. **Entropia come misura di possibilità future** (cf. paper §2.1 Causal Cones)
2. **Intelligenza come massimizzazione di futuri accessibili** (cf. paper §1.1)
3. **Vita come dissipazione di entropia interna** (cf. Book #2 §5.5 Dissipation maximization)

**Connessione con Wissner-Gross**: il post cita esplicitamente il concetto di "causal entropic force" introdotto da Wissner-Gross & Freer nel 2013 — la sorgente teorica diretta del programma Fractal AI.

**Stile**: scrittura accessibile, analogie concrete (le api nella sfera). Sergio era già allora un comunicatore efficace, ma non in linguaggio accademico standard. Probabilmente è uno dei motivi per cui il programma è stato sottovalutato dalla community ML mainstream.
