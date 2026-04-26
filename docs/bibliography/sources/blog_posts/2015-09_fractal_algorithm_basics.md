---
title: "Fractal Algorithm Basics"
author: Sergio Hernández Cerezo
date: 2015-09
url: https://entropicai.blogspot.com/2015/09/fractal-algorithm-basics.html
fetched: 2026-04-26
---

# Fractal Algorithm Basics

## Overview

This blog post introduces "Fractal Growth based algorithms," a computational approach grounded in thermodynamic principles. The author proposes that "intelligent behaviour" can be defined using entropy concepts, with fractalization improving performance by eliminating entropy calculations from AI systems.

## Core Concept: The Plant Pot Analogy

The foundational idea uses a metaphorical plant pot placed randomly in a field, attempting to locate the sunniest area. The system operates through:

- **Four directional seeds** pointing north, south, east, and west
- **Resource constraints** limiting active branches to 100 simultaneously
- **Growth-based decision making** where larger branches indicate favorable directions
- **Iterative repositioning** based on weighted branch measurements

After each growing season, branch widths translate into normalized coefficients, producing a weighted average that guides the pot's movement toward optimal conditions.

## Real-World Problem Translation

The algorithm generalizes to function optimization:
- The "field" becomes a function's domain
- "Sunny" conditions represent maximum function values
- The "pot" tracks the current best-known position
- "Branches" form the fractal structure for decision-making

## Plant Growth Mechanics

The fractal develops through daily iterations where:

- Branches collect energy (function values) at their positions
- Accumulated energy triggers bifurcation when capacity is reached
- Random branch pruning maintains the resource limit
- Worst-performing branches gradually diminish over time

This approach enables indefinite fractal growth while naturally weighting successful directions through branch density.

## Significance

The author argues these methods could transform complex problems, potentially reducing NP-complete problems toward polynomial or linear time complexity by leveraging nature's fractal patterns.

---

## Annotation (analisi storica)

**Importanza nel corpus**: questo post è il primo dove emerge la **metafora frattale** che darà il nome al programma. La "plant pot analogy" è un'immagine biologica potente per spiegare swarm intelligence + selezione darwiniana.

**Connessione con FMC**:
- "Branches" = walker
- "Branch width" = virtual reward (peso del walker)
- "Random branch pruning" = cloning step (i walker scarsi muoiono)
- "Bifurcation" = perturbation/exploration step

È **lo stesso algoritmo** del paper 2018, descritto con metafore botaniche invece che termodinamiche.

**Affermazione audace**: "potentially reducing NP-complete problems toward polynomial or linear time complexity". È una claim molto forte che merita scetticismo, ma indica l'ambizione del programma.

**Lezione di stile**: 3 anni prima di FMC, Sergio aveva già il **kernel concettuale completo** in metafora botanica. La progressione dal blog ai paper è prevalentemente di formalizzazione, non di scoperta. Le idee erano lì.
