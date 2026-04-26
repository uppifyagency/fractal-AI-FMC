---
title: "The New Intelligence Level 7"
author: Sergio Hernández Cerezo
date: 2014-03-25
url: http://entropicai.blogspot.com/2014/03/the-new-intelligence-level-7.html
fetched: 2026-04-26
---

# The New Intelligence Level 7

**Posted by Sergio Hernandez on Tuesday, March 25, 2014**

## Overview

This blog post introduces "Level 7" intelligence, described as an advancement in entropy-based AI algorithms. The author presents a comparison of different intelligence levels through a racing game demonstration featuring four karts using different algorithmic approaches.

## Key Points

**Problem Being Addressed:**
The author identifies the need for negative scoring in AI systems to enable genuine competitive intelligence capable of optimizing decisions and outperforming opponents.

**Intelligence Levels Compared:**

- **Level 3 (White kart):** Uses simple entropy formula k*Log(N), treating all futures equally. The author describes it as "irresponsible" and prone to getting stuck in difficult situations.

- **Level 5 (Yellow kart):** Incorporates squared distance in scoring calculations across multiple futures. It aligns with entropy definitions but lacks a clear theoretical framework for decision-making.

- **Level 6 (Orange kart):** A "strange mixture" formula designed to improve reaction speed in complex bifurcations, though less efficient overall.

- **Level 7 (Red kart):** The new approach subtracts the minimum score from all option scores before normalization, allowing negative values while preserving a normalized sum of 1.

## Distinguishing Feature

"ZeroedScores" ensures the lowest-scoring options register zero while maintaining relative proportions. This produces faster decision-making without violating entropy principles.

## Conclusion

Level 7 demonstrates superior performance, notably deciding to collide with a stuck obstacle to create space, demonstrating genuine adaptive behavior.

---

## Annotation (analisi storica)

**Importanza nel corpus**: questo è uno dei post più antichi di Sergio sulla scala dell'evoluzione dell'algoritmo. Mostra come — già nel 2014 — l'idea di intelligenza era operativizzata come "scoring di azioni con bilanciamento entropia/distanza/reward".

**Connessione con Book #1**: il "ZeroedScores" trick è un antesignano della funzione `relativize` del paper 2018 (§2.2.3). Stesso problema (reward arbitraria → dover comprimere/normalizzare), prima soluzione naïve (subtract min), poi soluzione raffinata (z-score + log/exp).

**Connessione con FMC**: i diversi "kart" sono già walker in nuce. Sergio sperimentava con diverse formule di scoring per vedere quale producesse migliore comportamento emergente.
