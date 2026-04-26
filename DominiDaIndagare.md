Mappa dei domini — verifica del need reale
I criteri (espliciti, non barattabili)
Per essere un buon target FMC + paper credibile servono tutti e quattro:

Sim < 10 ms/step (idealmente < 1 ms) — altrimenti N×M chiamate ti uccidono
OOD documentato — esiste un paper recente che dice "deep RL fallisce qui in OOD"
Benchmark vivo — codice mantenuto, citato da ≥10 paper/anno, riproducibile
Storia vendibile: o impatto industriale, o impatto scientifico, o un benchmark che la community usa come metro
Filtra brutalmente: la maggior parte dei "buoni candidati" muore su (3) o (4).

Survey completa con verdetto
Robotics/Control
Dominio	Sim	OOD	Bench	Verdetto
Drone obstacle avoidance (Flightmare/AirSim)	✓	✓	medio	buono — ma deep RL + sim2real è un fronte presidiato
Quadruped locomotion novel terrain	✓	✓	forte	scarta — MIT/ETH/Boston Dynamics dominano, FMC non ha story sim2real
Manipulation novel objects	✓	✓	fortissimo	scarta — RT-X / Open X-Embodiment hanno saturato l'agenda
Soft robot control	✓	parziale	debole	scarta — community troppo piccola
AUV / underwater	✓	✓	debolissimo	scarta — bench non standard
Logistics / OR
Dominio	Sim	OOD	Bench	Verdetto
Traffic micro-sim (SUMO/CityFlow)	✓	✓	medio	buono — già menzionato
Warehouse routing (RWARE)	✓	parziale	debole	scarta — niche
Ride-sharing dispatch	✓	✓	proprietario	scarta — dati chiusi
Container/bin packing	✓	✓	medio	OK — combinatorial niente di sexy
Emergency dispatch	✓	✓	nullo	scarta
Game-like / Combinatorial — qui c'è la miniera
Dominio	Sim	OOD	Bench	Verdetto
Procgen Benchmark	~1ms	⭐ esplicito train/test	⭐⭐⭐ ICML 2020	TOP
Crafter	~5ms	✓ hierarchical	⭐⭐ ~30 paper	TOP
MiniHack	~ms	✓ procedural	⭐⭐ FAIR/NeurIPS 2020	forte
NetHack Learning Env	~ms	✓	⭐⭐ competition	rischioso — solver dedicati duri da battere
Crafter-OOD variants	~5ms	✓	⭐	forte
Boxoban/Sokoban	~ms	parziale	medio	scarta — A*+PDB già SOTA
MicroRTS	~ms	✓	medio	OK — competizioni ne hanno
Pommerman	~ms	✓	medio	OK — multi-agent
GRF (Google Football) Academy	~10ms	✓	medio	OK ma DeepMind sopra
Sistemi / Networking
Dominio	Sim	OOD	Bench	Verdetto
CompilerGym (LLVM passes)	10-100ms	⭐ documentato	⭐⭐ FAIR/CGO 2022	TOP
Adaptive Bitrate Streaming (Park/Pensieve)	<1ms	⭐ esplicito	⭐⭐ MIT/NSDI 2017	forte
Datacenter scheduling (Park/Decima)	<1ms	✓	⭐⭐ MIT/SIGCOMM 2019	forte
Database query plan	<1ms	✓	medio	OK
TCP congestion control	<1ms	✓	debole	scarta — niche
Network packet routing (ns-3)	varia	✓	medio	OK
Cache replacement	nano	parziale	debole	scarta
Scientific / Discovery
Dominio	Sim	OOD	Bench	Verdetto
Molecular conformer search (RDKit/MMFF)	ms	✓ scaffold OOD	⭐⭐ Conformer-RL etc	forte
Retrosynthesis planning	ms (LLM judge)	✓	⭐⭐ USPTO bench	forte ma LLM-heavy
Drug docking (AutoDock Vina)	sec	parziale	medio	scarta sim lento
Protein design (Rosetta-Lite)	sec	✓	medio	scarta sim lento
Crystal structure prediction	sec	✓	medio	scarta sim lento
Material discovery (empirical pot.)	ms	✓	medio	OK
Quantum circuit compilation	ms	✓	⭐ in crescita	forte niche
Healthcare / Sequential decision
Dominio	Sim	OOD	Bench	Verdetto
Insulin dosing (UVa-Padova T1D)	ms	⭐ patient OOD	⭐⭐ FDA-recognized	forte impact
Sepsis treatment	offline only	⭐	controverso	scarta ethically
ICU vasopressor dosing	offline only	⭐	controverso	scarta
Mechanical ventilation	ms	✓	debole	OK ma niche
Finanza
Dominio	Sim	OOD	Bench	Verdetto
Market making	<1ms	⭐ regime change	medio (dati non pubblici)	scarta — bench non standard
Portfolio rebalancing	ms	✓	proprietario	scarta
Optimal execution VWAP/TWAP	ms	parziale	medio	OK
Option hedging w/ tx costs	ms	✓	⭐ Buehler et al.	OK
Energy / Smart grid
Dominio	Sim	OOD	Bench	Verdetto
EV charging coordination	ms	✓	medio	OK
Battery dispatch microgrid	ms	✓	medio	OK
Building HVAC (EnergyPlus)	sec	✓	medio	scarta sim lento
Frequency control grid	ms	✓	debole	scarta niche
I tre TOP — verifica del need
⭐ 1. Procgen Benchmark
Riferimento: Cobbe, Hesse, Hilton, Schulman, "Leveraging Procedural Generation to Benchmark Reinforcement Learning", ICML 2020, OpenAI. https://github.com/openai/procgen

Perché è il the benchmark per il pitch FMC:

16 giochi procedurali. Train su 200 livelli, test su distribuzione infinita di livelli mai visti
Gap train-test di PPO/Rainbow ≈ 50-60% (numero pubblico, citato in 50+ paper)
Tutti i metodi che provano a chiudere il gap (DAAC, PPG, IDAAC, mixreg, RAD, DrAC, …) sono training-time tricks
Un agente che non si addestra ha gap zero by construction. Lo argomento è elegante e immediato
Need verificato:

~10 paper/anno usano Procgen come bench
OpenAI lo mantiene
È citato in ogni survey su generalization in RL (es. Kirk et al. JAIR 2023 "A Survey of Generalisation in Deep RL" cita Procgen come benchmark canonico)
Rischi reali:

Alcuni giochi (Heist, Maze) richiedono memoria → FMC vanilla è memoryless → potresti perdere lì
Servono sim parallele perché 16 giochi × N seed × FMC è un po' costoso
Sim non è < 1ms su tutti, alcuni ~5ms — comunque OK
Pubblicabilità: workshop garantito (NeurIPS Generalization in RL workshop esiste). Main track possibile se i numeri reggono su ≥10 dei 16 giochi.

⭐ 2. Crafter
Riferimento: Hafner, "Benchmarking the Spectrum of Agent Capabilities", arXiv:2109.06780 (2021), poi adottato come bench da Dreamer-V3 (Hafner Nature 2025).

Perché è interessante:

Survival/crafting con 22 achievement gerarchici (collect-wood → make-pickaxe → mine-stone → …)
Reward sparse e composto (gerarchia di skill)
Procedural generation ✓
Numeri pubblici: PPO ~4.5/22, IMPALA ~7/22, DreamerV3 ~10/22 (SOTA)
Il "score Crafter" è una metrica monotonica: ogni achievement in più conta
Perché FMC è strutturalmente vincente qui:

Reward composta moltiplicativa (paper §2.2.2): se una achievement non è raggiunta, $R = 0$ → FMC scopre la gerarchia senza credit assignment learnt
Common Sense (α=0) tiene l'agente vivo (cibo/acqua/sonno) mentre esplora
Lo sciame può "dividere il lavoro" implicitamente (un walker scava, uno raccoglie, uno mangia)
Need verificato:

~30+ paper citano Crafter come benchmark
DreamerV3 (Hafner 2023, Nature 2025) lo usa come banco principale
ICLR/NeurIPS papers usano Crafter per testare hierarchical RL
Rischi:

Orizzonte lungo (10K+ step per game). FMC pianifica solo τ avanti, può non vedere "la pickaxe ti servirà tra 500 step"
Mitigazione: usare reward intrinseche (potential-based) sulle achievement parziali
Pubblicabilità: workshop solidissimo. Main track possibile se batti DreamerV3 anche solo su un subset.

⭐ 3. CompilerGym
Riferimento: Cummins et al., "CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research", CGO 2022 (Facebook AI). https://github.com/facebookresearch/CompilerGym

Perché è il "industry impact" pitch perfetto:

LLVM optimization sequence selection: scegliere quale pass applicare next
Reward = code size reduction o runtime improvement (entrambi misurabili)
Deep RL fallisce OOD: PPO trained su un sottoinsieme di programmi degrada su programmi mai visti — Cummins et al. lo documentano esplicitamente
Il valore in produzione è reale: ogni % di binary size = $$$ in distribuzione di app
Perché FMC è vincente:

Per ogni programma nuovo, FMC ri-pianifica online. Non c'è "OOD" perché non c'è "ID"
Sim = subprocess di LLVM con N pass applicati = 10-100 ms (marginale ma OK per offline use)
Discrete action space (≈ 100 LLVM pass)
Need verificato:

Mantenuto da Meta AI Research
~30 paper lo usano (Mantle, MLGO di Google, etc.)
Google ha messo in produzione MLGO (Trofin et al. 2021): la community vuole un metodo che generalizza meglio
Rischi:

Sim 10-100 ms è il limite alto del nostro budget. N=30, M=20 = 600 chiamate × 50 ms = 30 sec/decisione → solo offline, non interactive compilation
LLVM pass scheduling ha già strong baselines (Mantle, OpenTuner)
Pubblicabilità: MLSys, CGO, ML for Systems workshop. Industry-relevant è una bandiera che attrae review favorevoli.

I quattro "honorable mentions" che hanno una storia vera
4. Adaptive Bitrate Streaming — Pensieve / Park
Riferimenti:

Mao, Netravali, Alizadeh, "Neural Adaptive Video Streaming with Pensieve", SIGCOMM 2017 (MIT)
Mao et al., "Park: An Open Platform for Learning-Augmented Computer Systems", NeurIPS 2019
Pitch: ogni servizio video streaming sceglie la bitrate dei prossimi 4 secondi sulla base di throughput stimato. Pensieve ha mostrato deep RL > heuristics. Successivi paper hanno mostrato che Pensieve degrada su trace di rete OOD (es. 5G non visto in train).

Vantaggi FMC: simulator è un replay di trace → < 1 ms/step. State piccolo (~10 dim). Action discreta (~6 bitrate). MDP pulitissimo.

Need: niche ma con citazioni concrete (Pensieve ha ~1500 cit). Il problema è che il bench è dominato da una community sistemi (NSDI/SIGCOMM), non ML.

Pubblicabilità: ML4Sys workshop, Sigcomm CoNEXT.

5. Insulin dosing — UVa-Padova T1D Simulator
Riferimenti:

Man et al., "The UVA/PADOVA Type 1 Diabetes Simulator: New Features", J Diabetes Sci Technol 2014. FDA-accepted sostituto di trial preclinici
Fox et al., "Reinforcement Learning for Personalized Glucose Control in T1D", ML4H 2020
Pitch: ogni paziente T1D è una distribuzione diversa di parametri fisiologici. Deep RL trained su un cohort fallisce su pazienti out-of-cohort. FMC pianifica online sul modello del singolo paziente → adattamento istantaneo.

Vantaggi: simulator è ODE coupled, ms/step. Impatto medico chiaro.

Rischi: dominio medico ha review più severi su sicurezza. Servono garanzie hard (no hypoglycemia).

Pubblicabilità: ML4H (ML for Health) workshop, JBHI (Journal of Biomedical and Health Informatics). Citazioni: ~40 paper Pediatric Diabetes Care.

6. Molecular conformer search
Riferimento: Conformer-RL (Sridharan et al. 2022), GeomNet baselines. Bench: GEOM-DRUGS, MMFF94 force field via RDKit (~ms per energy eval).

Pitch: trovare conformer a bassa energia di una nuova molecola. Deep RL trained su scaffold X non generalizza a scaffold Y novel. FMC pianifica online sulla molecola specifica.

Need: chemoinformatics community attiva, MolNet/GEOM benchmark mantenuti. ~30 paper/anno.

Rischi: distance metric tra conformer è tricky (RMSD non-Euclidea). Potrebbe servire proprio quello che il Book #2 chiama "embedding learnt" → reintroduzione learning.

7. MiniHack
Riferimento: Samvelyan et al. "MiniHack the Planet", NeurIPS 2021 (FAIR).

Pitch: micro-task da NetHack, procedural, FAIR-mantenuto. OOD esplicito. Sim ms.

Rischi: già più presidiato (NLE Challenge 2021-23). FMC vanilla potrebbe essere sotto i symbolic-RL ibridi.

Le combinazioni che genererei
Se vuoi un singolo paper-shaped progetto la mia raccomandazione è duplice e dipende dall'angolo:

Opzione A — "FMC chiude il gap OOD by construction" (più pulito, più rischio)
→ Procgen come benchmark principale + Crafter come secondario.
Pitch: "Forward planning ha gap train-test = 0 by construction. Mostriamo che FMC è competitivo con SOTA in-distribution e domina in OOD."
Tempo: 2-3 mesi.
Rischio: se FMC non è competitivo su Maze/Heist (memoria), il paper si sbilancia.

Opzione B — "FMC come planner online dove deep RL è impratico" (più applicato, meno rischio)
→ Adaptive Bitrate Streaming (Park) come bench principale + Insulin dosing come secondario.
Pitch: "Quando il deployment richiede adaptive personalization e il deep RL retraining è non praticabile, FMC online dà X% migliore performance senza training."
Tempo: 1-2 mesi per Park; 3-4 per insulin.
Rischio: niche audiences.

Opzione C — "FMC come compiler heuristic" (industry-flavored)
→ CompilerGym + studio di scaling N×M vs program complexity.
Pitch: "Replacing trained heuristic with online planning gives more robust optimization across novel programs."
Tempo: 3-4 mesi.
Rischio: 50ms/step è il limite — devi vettorizzare.

