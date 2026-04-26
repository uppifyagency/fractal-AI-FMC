# 01 — Setup ambiente di esecuzione

**Goal verificabile**: arrivare a `python verify_install.py` che ritorna `OK ✓` per tutti gli step (import, ambiente Atari, smoke test FMC).

## Scelta strategica: due percorsi paralleli

Esistono due strade praticabili. Le proponiamo entrambe perché hanno tradeoff opposti:

| Path | Repo | Python | Difficoltà | Velocità | Pro | Contro |
|---|---|---|---|---|---|---|
| **A — Legacy** | `FractalAI_old` | 3.6-3.8 | bassa | lenta (CPU) | funziona out-of-the-box, codice del paper | deprecato, gym vecchio |
| **B — Modern** | `fragile` | ≥3.10 | media | veloce (GPU) | PyTorch, vettorizzato, GPU | più dipendenze (panel, holoviews, ray, plangym, shaolin) |

**Raccomandazione**: iniziare con **Path A** per uno smoke test in ~30 minuti, poi passare a **Path B** per i benchmark seri di [`03_atari_replication/`](../03_atari_replication/).

---

## Path A — `FractalAI_old` (smoke test rapido)

### Prerequisiti macOS

```bash
# Python 3.8 (ultimo supportato da gym[atari] vecchio)
brew install python@3.8 cmake boost boost-python sdl2 swig wget

# Atari ROM — necessarie per Atari (legali per uso non-commerciale dal 2022)
# Verranno installate via autorom dopo il pip install
```

### Installazione

```bash
cd repos/FractalAI_old
python3.8 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel setuptools

# Dipendenze del paper (commentate nel setup.py originale)
pip install numpy networkx jupyter ipython
pip install "gym==0.21.0"               # ultima versione compatibile
pip install "ale-py==0.7.5"             # Arcade Learning Environment
pip install "autorom[accept-rom-license]"
AutoROM --accept-license

pip install -e .
```

### Verifica

```bash
python -c "from fractalai.swarm import Swarm; print('Swarm OK')"
python -c "import gym; env = gym.make('MsPacman-v0'); print('Atari OK')"
```

### Demo notebook

```bash
jupyter notebook FMC_example.ipynb
```

---

## Path B — `fragile` (production grade)

### Prerequisiti

- **Python ≥ 3.10** (raccomandato 3.11)
- **uv** o **pip** moderni
- **Git LFS** (alcuni asset sono LFS)
- **CUDA** opzionale ma raccomandato per benchmark seri

### Installazione

```bash
cd repos/fragile
python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel setuptools

# Le dipendenze sono in pyproject.toml
pip install -e .

# Plangym per Atari (è il wrapper moderno usato da fragile)
pip install "plangym[atari]>=0.1.29"

# ROM Atari
pip install "autorom[accept-rom-license]"
AutoROM --accept-license
```

### Verifica

Vedi [`verify_install.py`](verify_install.py) e [`verify_install.sh`](verify_install.sh).

```bash
bash verify_install.sh
```

---

## Matrice di compatibilità nota

| OS | Python | Path A | Path B | Note |
|---|---|---|---|---|
| macOS arm64 (M1+) | 3.8 | ⚠️ pyglet/SDL2 issues | ⚠️ torch needs MPS | usare conda-forge |
| macOS x86_64 | 3.8 / 3.11 | ✅ | ✅ | testato |
| Linux x86_64 | 3.10/3.11 | ✅ | ✅ | path consigliato |
| Windows | qualsiasi | ❌ | ⚠️ | usare WSL2 |

## Troubleshooting

| Errore | Causa | Soluzione |
|---|---|---|
| `ModuleNotFoundError: cv2` | OpenCV non installato | `pip install opencv-python` |
| `ROM not found: ms_pacman` | Atari ROM mancanti | `AutoROM --accept-license` |
| `gym.error.NameNotFound: Environment ALE/MsPacman not found` | versione gym/ale incompatibile | downgrade gym a 0.21 o usare gymnasium |
| `ImportError: shaolin` | pacchetto interno FragileTech | `pip install git+https://github.com/FragileTech/shaolin` |
| `RuntimeError: CUDA out of memory` | walker count troppo alto su GPU | abbassa `n_walkers` a 100 |

## Output atteso al successo

```
$ bash verify_install.sh
[1/5] Python version......... 3.11.7 ✓
[2/5] Importing fragile...... 0.1.x ✓
[3/5] Importing torch........ 2.x.x ✓
[4/5] Importing plangym..... 0.1.29 ✓
[5/5] Smoke test FMC........ Episode reward: 670.0, samples: 4123 ✓

OK ✓ — ambiente pronto per replication
```
