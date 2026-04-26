---
title: "Solved Atari Games"
author: Sergio Hernández Cerezo
date: 2017-06
url: http://entropicai.blogspot.com/2017/06/solved-atari-games.html
fetched: 2026-04-26
---

# Solved Atari Games

## Overview

This blog post documents an AI system based on thermodynamic principles that has successfully solved multiple Atari games from OpenAI's Gym environment. The approach uses entropy-based calculations and fractal optimization rather than traditional machine learning.

## Core Methodology

The researchers base their work on "Causal Entropic Forces" by Alexander Wissner-Gross and apply concepts from the G.A.S. algorithm. A key distinction is that their system "is not actually learning in any way," meaning each game is independent and requires no prior experience.

## Solved Environments

### Completed Games (100+ plays with official scoring)

1. **MsPacman-ram-v0**: Achieved approximately 11.5k average score versus 9.1k for the second-best algorithm (1.2x improvement). The main challenge involved accounting for a 15-frame animation delay when Pacman dies.

2. **Qbert-ram-v0**: Scored 18.4k average versus 4.2k for competitors (4.3x improvement). Frame-scanning was optimized to every other frame due to action animation delays.

### Incomplete Games (fewer than 100 plays)

3. **MsPacman-v0**: Preliminary results (~9k) showed 1.4x improvement over baseline (6.3k).

4. **Tennis-ram-v0**: Single-game result (~8) vastly outperformed the closest competitor (0.01), yielding an 800x ratio.

5. **VideoPinball-ram-v0**: Average score of ~500k across 21 games versus 28k baseline (17.8x improvement).

6. **CartPole-v0**: Achieved the theoretical maximum with 0.0 score (indicating optimal solving on first 100 games).

## Technical Insights

The fractal approach reportedly eliminates entropy calculations from the AI runtime while improving performance. The authors note that computational power significantly influences results—additional processing could make certain games nearly unbeatable.

## Limitations and Future Work

The blog post indicates testing remains incomplete due to computational constraints, with the author humorously noting their laptop's energy consumption. They committed to updating the list as more environments are solved.

---

## Annotation (analisi storica)

**Importanza nel corpus**: questo post documenta i **risultati Atari un anno prima** del paper 1807.01081 (luglio 2018). Mostra che la pipeline funzionava già nel giugno 2017 e Sergio stava iterando empiricamente.

**Numeri rilevanti** (giugno 2017):
- MsPacman: 11.5k (vs paper 2018 V5: 999,990 — **3 ordini di grandezza in più dopo l'ottimizzazione**)
- Qbert: 18.4k (vs paper V5: 999,975 — **idem**)
- VideoPinball: 500k (vs paper V5: 999,999)

I numeri del 2017 erano già SoTA, ma il paper 2018 li ha portati ai bug-induced limit dei giochi (il "1M bug" del paper §5.1.1).

**Lezione**: c'è stata una **iterazione algoritmica intensa** tra giugno 2017 e luglio 2018. Il paper finale rappresenta il punto di massima ottimizzazione, non il primo successo. Questo è coerente con il pattern di pensiero sperimentale-prima-di-formale tipico del programma.

**Connessione con la mia replica**: il mio Boxing 96/100 con `fmc_minimal.py` è in questa fascia di "primi successi" — non massimamente ottimizzato, ma sostanzialmente vicino al risultato. Per arrivare al 100/100 servirebbe lavoro di ottimizzazione paragonabile a quello fatto da Sergio tra 2017 e 2018.
