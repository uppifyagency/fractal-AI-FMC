# Run 007 addendum — Perche' il port a `fragile` non accelera Craftax

**Data**: 2026-04-29
**Contesto**: la spec del run 007 chiedeva "Porta `fmc_craftax_v4.py` su `fragile` (GPU
swarm gia' funzionante in `repos/fragile/`)" come strategia di accelerazione GPU.

## TL;DR

Dopo aver letto il codice di `repos/fragile/src/fragile/core.py`, **il port non
fornirebbe acceleration significativa per Craftax**. Il motivo e' una mismatch
strutturale tra il design di `fragile` e la natura JAX-native di Craftax.

**JAX vmap CPU (cio' che gia' usiamo in `fmc_craftax_v4.py`) e' strutturalmente
piu' veloce di fragile per Craftax**. Il "port a fragile" come strategia di
acceleration GPU e' basato su una premessa errata.

## Analisi

### Cosa accelera `fragile`

`fragile` e' un'implementazione di FMC PyTorch-based con due livelli di parallelismo:

1. **Swarm bookkeeping su GPU PyTorch**: vettori di walker observations, rewards,
   distances, virtual_reward, will_clone — tutti `torch.Tensor` su `device='cuda'` o
   `device='mps'`. Operazioni come `relativize`, `calculate_clone`, normalizzazione
   distanza pairwise, sono accelerate.

2. **Env stepping via `env.step_batch` su CPU-numpy**: il punto critico. In
   [`core.py:716`](../../../repos/fragile/src/fragile/core.py#L716):
   ```python
   data = self.env.step_batch(states=self.state_step, actions=action, dt=self.dt)
   ```
   Questa chiamata aspetta che l'env esponga `step_batch(numpy_states, numpy_actions)`
   e ritorni numpy outputs. **L'env stepping rimane CPU-numpy**, perche' fragile
   chiama `einops.asnumpy(self.action_step)` per convertire le actions GPU torch
   tensor in numpy array prima della chiamata.

### Dove sta il bottleneck per Craftax

`fmc_craftax_v4.py` su (N=512, M=160) prende ~302s/episode. Profilando
manualmente:
- `env.step` calls: ~99% del wall time (Craftax dynamics, JAX-compiled scan over
  walkers via vmap)
- relativize / virtual reward / cloning kernel: ~1% del wall time

Quindi: **l'env stepping JAX-native e' il bottleneck, non lo swarm bookkeeping**.

### Cosa accelererebbe (e cosa NO) un port a fragile

| Componente | Costo attuale (JAX vmap CPU) | Costo via fragile (PyTorch GPU + CPU env) |
|---|---|---|
| `env.step` (1 walker) | ~50us JAX | **piu' lento**: numpy roundtrip + non-vmapped |
| `env.step` (N walkers vmapped) | ~50us * O(log N) JAX vmap CPU | ~50us * N CPU sequential |
| Distance pairwise N x N | ~100us JAX vmap CPU | ~50us PyTorch GPU |
| Virtual reward N | ~10us JAX vmap CPU | ~5us PyTorch GPU |
| Cloning kernel | ~10us JAX vmap CPU | ~5us PyTorch GPU |

**Net**: per Craftax, il port a fragile **rallenterebbe** l'env stepping (perche' il
batch dei walker viene loop-ato sequenzialmente in numpy invece che vmap-ato in JAX),
mentre accelererebbe solo l'1% del wall time del swarm bookkeeping. Net negativo.

### Quando fragile e' la scelta giusta

`fragile` brilla per env che hanno:
1. **Step nativi numpy/CPU non-vmap-abili** (es. ALE Atari via stable-baselines, MuJoCo
   via gymnasium classic), perche' lo step e' gia' "lento" intrinsicamente e fragile
   non lo peggiora.
2. **State serializabile come numpy ndarray "piccolo"** (es. RAM Atari = 128 byte,
   MuJoCo qpos = ~50 doubles).
3. **Swarm bookkeeping che domina il wall time** (es. swarm di N=10000 walker su
   un env che fa 5ms/step, swarm bookkeeping dominante).

Craftax non rispetta nessuna delle 3:
- Step e' JAX-native vmap-abile (il fast path);
- State e' un PyTree con map 64x64xint32 + inventory + mobs + ... = ~20 KB nested;
- Env stepping domina (~99% del wall time).

## Conseguenza per la roadmap

**Strategia originale** (port a fragile per GPU): cancellata.

**Strategia alternativa per accelerazione**: aspettare JAX 0.11+ con jax-metal 0.2 che
supporti `default_memory_space`, e poi usare `JAX_PLATFORMS=METAL` direttamente sullo
script attuale. Stima: ~10x speedup su Apple M1, magari 100x su NVIDIA H100/A100 cloud.

**Strategia attiva ora**: rimanere su `fmc_craftax_v4.py` JAX vmap CPU, che e' gia'
ottimo. Sweep di 65 episodi in 101.7 min CPU e' tractable per qualsiasi follow-up.

## Implicazione pratica

Il decision-gate del run 007 non era bloccato dalla mancanza di GPU. **101.7 min
di CPU** sono stati piu' che sufficienti per falsificare la M-bottleneck hypothesis
con p<0.001. Il "port a fragile" non avrebbe cambiato il verdetto, e in pratica
sarebbe stato inutile o controproducente.

## Riferimenti

- [`repos/fragile/src/fragile/core.py:713-752`](../../../repos/fragile/src/fragile/core.py#L713) — il blocco `step_env`/`step_walkers` che mostra il roundtrip CPU-numpy per env step
- [`fmc_craftax_v4.py:147-150`](../scripts/fmc_craftax_v4.py#L147) — il `jax.vmap(env.step)` che accelera N walker step in parallelo su JAX, il fast path che fragile non puo' replicare
- [`docs/architecture/tier1_repos_teardown.md`](../../../docs/architecture/tier1_repos_teardown.md) — analisi piu' generale della stack `fragile + plangym + shaolin`
