# Corpus completo Hernández-Cerezo & Duran-Ballester (2014-2026)

> *Tutte le pubblicazioni, draft, blog e codebase rilevanti al programma Fractal AI, raccolte e organizzate cronologicamente.*

## ⚡ Quick access — file scaricati localmente

Tutti i materiali sono stati scaricati e archiviati in [`sources/`](sources/):

| Categoria | Path | Contenuto |
|---|---|---|
| 📄 Paper PDF | [`sources/papers/`](sources/papers/) | 4 PDF (3 arXiv + 1 Entropy journal) |
| 📚 Drafts | [`sources/books/`](sources/books/) | Book #2, Hives, Fractal Memory Slide |
| 📝 Blog posts | [`sources/blog_posts/`](sources/blog_posts/) | 8 post archiviati con annotazioni + INDEX |
| 🔗 External refs | [`sources/external_refs/`](sources/external_refs/) | Riservato per future espansioni |

**Autori principali**:
- **Sergio Hernández Cerezo** — CEO HCSoft, fondatore Fragile Technologies. Matematico (Univ. Valencia). Twitter: [@EntropyFarmer](https://twitter.com/EntropyFarmer). Email: sergio@hcsoft.net.
- **Guillem Duran Ballester** — Co-fondatore Fragile Technologies. Twitter: [@Miau_DB](https://twitter.com/Miau_DB).

**Affiliazioni**:
- HCSoft Programación (Spagna)
- FragileTech / Fragile Technologies
- Universidad de Elche (collaborazioni con José M. Amigó)

---

## 1. Cronologia completa

| Anno | Tipo | Titolo | Note |
|---|---|---|---|
| 2014–2019 | Blog (multipost) | [Entropic and Fractal Intelligence](http://entropicai.blogspot.com/) | Lab notebook pubblico, 5+ anni di posts |
| 2017 | arXiv | [General Algorithmic Search](https://arxiv.org/abs/1705.08691) → 📁 [`PDF locale`](sources/papers/2017_general_algorithmic_search_1705.08691.pdf) | **Antesignano**: meta-euristica swarm globale |
| 2018 | Journal | [A Brief Review of Generalized Entropies](https://www.mdpi.com/1099-4300/20/11/813) → 📁 [`PDF locale`](sources/papers/2018_brief_review_generalized_entropies.pdf) | Entropy Vol. 20 No. 11. Co-autori: Amigó, Balogh. **236 citazioni** |
| 2018 | arXiv (Book #1, V1) | [Fractal AI: A Fragile Theory of Intelligence](https://arxiv.org/abs/1803.05049v1) | Prima versione |
| 2018 | arXiv | [Solving Atari Games Using Fractals And Entropy](https://arxiv.org/abs/1807.01081) → 📁 [`PDF locale`](sources/papers/2018_solving_atari_1807.01081.pdf) | Co-autore: Spiros Baxevanakis |
| 2018-2019 | arXiv (V2-V4) | Fractal AI revisions | 4 revisioni in 18 mesi |
| 2020 | arXiv (Book #1, V5) | [Fractal AI V5 (final)](https://arxiv.org/abs/1803.05049v5) → 📁 [`PDF locale`](sources/papers/2020_fractal_ai_v5_1803.05049.pdf) | **57 pagine, versione canonica** |
| 2020 | Draft privato | **Fractal AI Book #2: AGI Structure** (V0.2) → 📁 [`MD locale`](sources/books/2020_book2_agi_structure.md) | non pubblicato |
| 2020 | Draft privato | **Honey Badger meets Fractal AI Hives** → 📁 [`MD locale`](sources/books/2020_hives_badger_meets_fractal_ai.md) | non pubblicato |
| 2020 | Draft privato (slide) | **Fractal Memory: Hybrids for Neural Networks** → 📁 [`MD locale`](sources/books/2020_fractal_memory_slides.md) | non pubblicato |
| 2021 | Book chapter | [Physics-Inspired Swarm Optimization: The General Algorithmic Search](https://www.worldscientific.com/) | World Scientific Series on Nonlinear Science. **Pubblicazione formale del lavoro 2017** |
| 2023 | arXiv | [Latent Diffusion Models for Histopathology](https://arxiv.org/abs/2312.09792) | Duran-Ballester come co-autore. **Off-topic** (medical imaging) ma rilevante per la traiettoria di carriera |
| 2024-2026 | Codebase + docs | [`fragile-rl`](https://github.com/FragileTech/fragile-rl) + Fragile Mechanics book | Successore canonico di Book #2 |

---

## 2. Pubblicazioni accademiche dettagliate

### 2.1 General Algorithmic Search (2017)

> **arXiv: [1705.08691](https://arxiv.org/abs/1705.08691)**
> Sergio Hernández, Guillem Duran, José M. Amigó

**Posizione nel corpus**: il **predecessore diretto** di Fractal AI. Introduce GAS (General Algorithmic Search), una meta-euristica swarm-based per ottimizzazione globale.

**Risultati**:
- Confrontato con Basin Hopping, Cuckoo Search, Differential Evolution su 31 funzioni di test
- GAS supera gli altri *especially in concurrent optimization* (più run con seed diversi)

**Importanza**: già qui c'è l'idea-cardine — uno sciame di agenti che evolve verso massimi globali, con balance esplorazione/sfruttamento. Il salto a FMC del 2018 è "applicare GAS a planning invece che optimization".

### 2.2 A Brief Review of Generalized Entropies (2018)

> **Entropy 20(11):813**, [DOI: 10.3390/e20110813](https://doi.org/10.3390/e20110813)
> José M. Amigó, Sámuel G. Balogh, **Sergio Hernández**

**Posizione nel corpus**: il **fondamento teorico**. Review accademica delle entropie generalizzate (Tsallis, Rényi, Hanel-Thurner, ecc.) che sostiene matematicamente il framework di Fractal AI.

**Citazioni**: **236** (la più citata del Sergio).

**Concetto chiave**: gli assiomi di Shannon-Khinchin sono violati da sistemi non-ergodici/long-range. Fractal AI usa questo formalismo per giustificare l'uso di reward composte non-additive.

### 2.3 Fractal AI: A Fragile Theory of Intelligence (2018-2020)

> **arXiv: [1803.05049](https://arxiv.org/abs/1803.05049)**
> Sergio Hernández Cerezo, Guillem Duran Ballester
> 5 versioni: V1 (2018-03), V2 (2018-06), V3 (2018-07), V4 (2019-12), V5 (2020-07)

**Il paper canonico** del programma. 57 pagine in V5. È quello che ho analizzato in [`ANALISIS.md`](../../ANALISIS.md) e che ho replicato empiricamente con [`fmc_minimal.py`](../../work/03_atari_replication/scripts/fmc_minimal.py).

**Citato come "Best of the Physics arXiv"** dal MIT Technology Review nella settimana di pubblicazione V1.

### 2.4 Solving Atari Games Using Fractals And Entropy (2018)

> **arXiv: [1807.01081](https://arxiv.org/abs/1807.01081)**
> Sergio Hernández Cerezo, Guillem Duran Ballester, **Spiros Baxevanakis**

**Posizione nel corpus**: il **paper sperimentale companion** del 1803.05049. Mentre 1803 è teoria + algoritmo, questo è prevalentemente experiments + benchmarks.

**Risultato chiave**: FMC batte MCTS UCT su Atari usando **<1000 sample/azione vs 3 milioni**. È il numero che ha attratto attenzione iniziale.

### 2.5 Physics-Inspired Swarm Optimization (2021)

> **World Scientific Series on Nonlinear Science**
> Multipli co-autori (Hernández, Duran, Amigó + altri)

**Posizione nel corpus**: la **versione formale ed estesa** del paper GAS del 2017. Pubblicazione book chapter, peer-reviewed.

**Importanza**: dimostra che gli autori **possono** pubblicare in venue tradizionali. La scelta di non pubblicare il Book #2 e Hives in venue formali è quindi una **decisione strategica**, non un'incapacità.

---

## 3. Drafts privati (in nostro possesso)

Tre documenti non pubblicati ufficialmente che il user ha condiviso:

### 3.1 Fractal AI Book #2: AGI Structure (V0.2)

[`Fractal Book.md`](../../Fractal%20Book.md) — il seguito del paper canonico. Analizza le funzioni esterne di FMC e propone l'architettura **Badger** per integrare learning + planning + reward.

**Contenuti chiave**:
- §2: Revisiting FMC (external functions analysis)
- §3: The Badger Structure
- §4: Building intuition (esempi biologici)
- §5: Entropic first principles in AGI (9 principi: 2nd law, least action, CEF, Fractal AI principle, dissipation, FEP, empowerment, surprise minimization, Kolmogorov complexity)
- §6: Research directions (consciousness, abstract reasoning, free lunch theorem)

Vedi [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](../../work/02_deep_dives/06_book2_badger_fractal_memory.md) per analisi dettagliata.

### 3.2 Honey Badger meets Fractal AI Hives

[`2020 Fractal.md`](../../2020%20Fractal.md) — la **specifica operativa** del Book #2 con pseudocodice Python. Definisce 5+ livelli (Outer/L4/L3/L2/L1/Expert/Walker), l'idea di "Learning as structural collapse", e la procedura di message passing tra livelli.

### 3.3 Fractal Memory: Hybrids for Neural Networks

[`2020 Fractal Slide.md`](../../2020%20Fractal%20Slide.md) — slide deck che applica il principio Fractal AI **dentro** le reti neurali:
- Dataset come Fractal Memory (curriculum learning automatico)
- Sinapsi come Fractal Memory (self-pruning)
- NN come Fractal Memory (multi-task con specializzazione automatica)

Plus la "Wigner reward" $R = \pi/2 \cdot x \cdot e^{-\pi x^2/4}$ come distribuzione ottimale di loss.

---

## 4. Lab notebook pubblico: Entropic and Fractal Intelligence

> **Blog**: http://entropicai.blogspot.com/
> **Mantenuto da**: Sergio Hernández Cerezo
> **Anni attivi**: 2014-2019 (ultimo post 2019)
> **Posts**: 50+ (esplorazione)

**Importanza**: questo è il "lab notebook pubblico" dove Sergio ha sviluppato le idee **prima** dei paper formali. Per chi vuole capire la genesi delle idee, è una miniera.

**Posts archiviati localmente** (8) — vedi [`sources/blog_posts/INDEX.md`](sources/blog_posts/INDEX.md):

- 📁 [`2014-03_intelligence_level_7.md`](sources/blog_posts/2014-03_intelligence_level_7.md) — antesignano `relativize`
- 📁 [`2014-03_rocket_in_cave.md`](sources/blog_posts/2014-03_rocket_in_cave.md) — anticipa §5.2 paper
- 📁 [`2014-03_the_entropy.md`](sources/blog_posts/2014-03_the_entropy.md) — foundation principles
- 📁 [`2015-05_fractal_function_optimization.md`](sources/blog_posts/2015-05_fractal_function_optimization.md) — prototipo GAS
- 📁 [`2015-09_fractal_algorithm_basics.md`](sources/blog_posts/2015-09_fractal_algorithm_basics.md) — plant pot analogy
- 📁 [`2015-12_fractal_ai_collaboration.md`](sources/blog_posts/2015-12_fractal_ai_collaboration.md) — **Octopus = Badger Structure**
- 📁 [`2016-04_pareto_frontiers.md`](sources/blog_posts/2016-04_pareto_frontiers.md) — single-objective philosophy
- 📁 [`2017-06_solved_atari_games.md`](sources/blog_posts/2017-06_solved_atari_games.md) — risultati pre-paper 2018
- 📁 [`2017-07_retrocausality_and_ai.md`](sources/blog_posts/2017-07_retrocausality_and_ai.md) — speculativo, antesignano Lorentzian memory

---

## 5. YouTube channel

> **Playlist**: https://www.youtube.com/playlist?list=PLEXwXLT-a6beFPzal3OznPQC0pieccAle

Video dimostrativi degli esperimenti, citati anche nel paper §5.2:
- [Solving the rocket task](https://youtu.be/HLbThk624jI)
- [Visualizing the decision process](https://youtu.be/cyibNzyU4ug)

---

## 6. Codebase / Software

| Repository | Anni | Stato | Scopo |
|---|---|---|---|
| [FragileTech/FractalAI](https://github.com/FragileTech/FractalAI) | 2018-2020 | Deprecato | Riferimento implementazione paper #1 (Python/NumPy) |
| [FragileTech/fragile](https://github.com/FragileTech/fragile) | 2020-presente | Attivo | Framework moderno PyTorch/GPU |
| [FragileTech/fragile-rl](https://github.com/FragileTech/fragile-rl) | 2024-2026 | Attivo | Successore Book #2 → Fragile Mechanics |
| [FragileTech/dockerfiles](https://github.com/FragileTech/dockerfiles) | 2018-2022 | Mantenimento | Docker images per CI/CD |
| [Guillemdb/FractalAI](https://github.com/Guillemdb/FractalAI) | 2018 | Mirror | Personal mirror di Duran-Ballester |
| [justindujardin/fragile](https://github.com/justindujardin/fragile) | — | Fork esterno | Fork community |

---

## 7. Mappa concettuale del corpus

```
                    [GAS 2017]
                       │
         ┌─────────────┴──────────────┐
         ↓                             ↓
[Generalized Entropies 2018]    [Fractal AI #1 2018-2020]
                                       │
                                       │ ←─── [FMC Atari paper 2018]
                                       ↓
                                [Fractal AI #2 2020 (draft)]
                                       │
                       ┌───────────────┼───────────────┐
                       ↓               ↓               ↓
                [Hives 2020]    [FM Slide 2020]  [Physics Swarm 2021]
                       │               │
                       └───────┬───────┘
                               ↓
                       [Fragile Mechanics 2024-2026]
                       (`fragile-rl` codebase + book)

LEGENDA:
  ───  pubblicato (peer-reviewed o arXiv)
  ─ ─  draft privato
```

---

## 8. Cosa ci manca (gap nel corpus)

Riconosciute lacune nel materiale a nostra disposizione:

1. **Fragile Mechanics book completo** — `fragile-rl/docs/source/1_agent/` ha ~10 parti, ne abbiamo letto solo l'introduzione. **Da leggere**: ~50 capitoli di matematica gauge, geometric DL, Lorentzian memory.

2. **Posts blog completi** — abbiamo i titoli, non i contenuti integrali.

3. **Documentazione interna `fragile`** — il codebase ha `docs/` con jupyter book, parzialmente esplorato.

4. **Tesi/talk video non transcritti** — la playlist YouTube ha video sostanziali, mai trascritti.

5. **Riferimenti incrociati con altri programmi AGI** — non abbiamo ancora cercato come questo corpus dialoga con: Yampolskiy, Schmidhuber, Hutter, Friston, Goertzel.

---

## 9. Contatti

Per estendere il corpus, possibili azioni:

1. **Email diretto a Sergio** (sergio@hcsoft.net) chiedendo se Book #2 ha versioni successive a V0.2
2. **Twitter outreach** ai due autori — la community è piccola, sono raggiungibili
3. **Fragile Tech website** — http://www.hcsoft.net/ — potrebbe avere risorse
4. **Inviti come reviewer/citation** sui paper futuri

---

*Ultimo aggiornamento: 2026-04-26. Mantenuto durante lo studio del programma Fractal AI.*
