---
title: "Fractal Function Optimization"
author: Sergio Hernández Cerezo
date: 2015-05-05
url: http://entropicai.blogspot.com/2015/05/fractal-function-optimization.html
fetched: 2026-04-26
---

# Fractal Function Optimization

## Main Content

This blog post discusses a fractal-based algorithm designed for function optimization, presented by Sergio Hernandez on May 5, 2015.

### Core Concept

The author describes a fractal algorithm that functions as a "maximizing plant," capable of locating a function's global maximum and then conducting deeper searches for improved solutions. The algorithm exhibits distinctive characteristics:

- **Function Flexibility**: Works with discontinuous functions, not requiring differentiability
- **Dimensional Versatility**: Operates across any number of dimensions
- **Noise Tolerance**: Can optimize even random noise to some degree

### Key Features

The post highlights that "most functions used to test those optimizing algorithms are designed for continuous functions, and the best algorithms also need the function to be diferentiable one or two times." In contrast, this fractal approach removes such constraints.

### Convergence Performance

The author demonstrates convergence through visual evidence rather than formal proof, acknowledging this limitation: "I can not formally prove it in any way, that is the naked truth (fractals are not so nice at proving things)."

A video shows 400 different starting positions converging toward the global optimum, with blue dots indicating slower growth and yellow dots showing more aggressive approaches. Notably, the algorithm parallelizes effectively.

### Practical Application

The author mentions benchmarking results were added in a subsequent post about the algorithm's real-world performance.

---

## Annotation (analisi storica)

**Importanza nel corpus**: questo è il **prototipo del paper "General Algorithmic Search" del 2017** (arXiv:1705.08691). Mostra che gli autori avevano la formulazione "swarm di walker per ottimizzazione globale" già 2 anni prima della pubblicazione.

**Insight chiave**: Sergio ammette esplicitamente l'**irrigorosità** dei suoi metodi ("fractals are not so nice at proving things"). Questa onestà è rara e va apprezzata. Il programma Fractal AI ha sempre privilegiato la dimostrazione **empirica** rispetto a quella formale — funziona, e poi si vedrà perché.

**Connessione moderna**: l'idea di "maximizing plant" è strutturalmente identica al CMA-ES e ad altri evolutionary strategies. Quello che FMC aggiungerà nel 2018 è applicare lo stesso pattern al **planning** invece che all'ottimizzazione di una funzione statica.
