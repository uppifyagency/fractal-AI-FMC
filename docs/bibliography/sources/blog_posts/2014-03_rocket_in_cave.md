---
title: "Driving a Rocket Inside a Cave!"
author: Sergio Hernández Cerezo
date: 2014-03-19
url: https://entropicai.blogspot.com/2014/03/driving-rocket-inside-cave.html
fetched: 2026-04-26
---

# Driving a Rocket Inside a Cave!

**Posted by Sergio Hernandez on Wednesday, March 19, 2014**

## Overview

Version 0.7 of the software introduced a generalized "Player2D" base class, enabling developers to create new vehicle types with minimal code. The author tested this by implementing a rocket that navigates a circuit as if it were a vertical cave with downward gravity.

## AI Behavior Observations

The AI attempts to maximize entropy, causing the rocket to accelerate aggressively. Interestingly, the rocket frequently spins to hover before deciding direction, as "the quickiest it rotates, the more future options it has." This spinning behavior demonstrates how the AI evaluates options that preserve flexibility.

When comparing the rocket to a traditional kart on the same circuit, distinct behavioral differences emerge. The author noted: "Watching the rocket fly out of the track because it tried to get a drop so blindly it didn't noticed that, after picking the drop, the crash with the circuit fences was inevitable."

## The Survival Problem

A critical issue surfaced: the AI lacks self-preservation instincts. Crashing scores zero rather than negative, leaving the AI unmotivated to avoid fatal collisions. The author proposed adding a survival goal to both vehicles.

## Proposed Solution

The current system uses positive scoring with reduction coefficients (0-1). The author plans to implement true negative scoring and a new goal: "try not to die in the next N seconds." When death occurs before N seconds elapse, the scoring formula would be:

**Score = Log((0.001+T)/N)**

Where T represents elapsed time. This ensures Log(1)=0 at safety threshold and approaches large negative values as death approaches immediately.

---

## Annotation (analisi storica)

**Importanza nel corpus**: questo è il **primo prototipo** dell'esperimento del razzo che apparirà 6 anni dopo nel paper §5.2 ("Flying a chaotic attractor"). La continuità intellettuale è impressionante.

**Insight chiave**: già nel 2014 Sergio osservava il comportamento "Common Sense" — il razzo *spinning to hover before deciding* perché ruotare lascia accesso a più futuri. Questa è esattamente la modalità α=0 del paper 2020. **L'idea era operativa 6 anni prima della formulazione formale.**

**Lezione metodologica**: gli AI senza penalità per la morte non sviluppano self-preservation. È un'osservazione che oggi sembra banale, ma nel 2014 (pre-RLHF, pre-GPT) era controintuitiva. Il post anticipa il problema della *reward shaping* che dominerà l'RL accademico.
