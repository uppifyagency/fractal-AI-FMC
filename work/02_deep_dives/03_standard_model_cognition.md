# Deep Dive 03 — Standard Model of Cognition (Fragile Mechanics)

> **Stato**: outline. Richiede studio approfondito di [`repos/fragile-rl/docs/source/1_agent/08_multiagent/`](../../repos/fragile-rl/docs/source/1_agent/) prima di poter essere completato.

## Tesi

> *Fragile Mechanics propone che il gruppo di simmetria $G_{\mathrm{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ emerga inevitabilmente da tre principi di invarianza dell'agente cognitivo, in analogia stretta con il Standard Model della fisica delle particelle.*

## Le tre invarianze (da espandere)

| Invarianza | Gruppo | Campo gauge | Analogia fisica |
|---|---|---|---|
| Utility phase | $U(1)_Y$ | $B_\mu$ (Opportunity) | Hypercharge |
| Sensor-motor chirality | $SU(2)_L$ | $W_\mu$ (Error) | Weak isospin |
| Feature basis freedom | $SU(N_f)_C$ | $G_\mu$ (Binding) | Color (gluon) |

## Outline delle sezioni da scrivere

1. **Cosa è una "invarianza" per un agente** — definizione operativa: trasformazione del modello che lascia invariato il comportamento
2. **Derivazione del primo gauge field**: $U(1)_Y$ da invariance phase del valore
3. **Sensor-motor chirality**: distinzione tra sensori (input) e motori (output) che rompe la simmetria parità
4. **Feature basis freedom**: arbitrarietà della rappresentazione interna delle feature
5. **Anomalie**: vincoli di consistenza che fissano le costanti di accoppiamento
6. **Parameter Space Sieve**: i 6 parametri fondamentali $(c_{\mathrm{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ derivati come soluzioni di vincoli
7. **Critica**: è una vera teoria o un'analogia formale? Test empirici proposti

## Riferimenti chiave (interni a fragile-rl)

- [`docs/source/1_agent/08_multiagent/01_standard_model.md`](../../repos/fragile-rl/docs/source/1_agent/08_multiagent/) (da leggere)
- [`docs/source/1_agent/08_multiagent/02_parameter_sieve.md`](../../repos/fragile-rl/docs/source/1_agent/08_multiagent/) (da leggere)
- [`src/fragile/layers/jump_operator.py`](../../repos/fragile-rl/src/fragile/) (covariant operators)
- [`src/fragile/agent.py`](../../repos/fragile-rl/src/fragile/agent.py) (FragileAgent stack)

## Critiche pre-emptive

Prima di scrivere il deep dive, vanno esaminate le obiezioni naturali:

1. **È un'analogia o una teoria?** Il Standard Model fisico ha *predizioni* (es. massa Higgs). Il SM cognitivo ha predizioni paragonabili?
2. **Falsificabilità**: quali esperimenti potrebbero refutare questo schema?
3. **Parsimonia**: il rasoio di Occam favorisce questo formalismo o un approccio più semplice?
4. **Convergenza con altre teorie unificate** (Friston FEP, Schmidhuber compression, Hutter AIXI)

---

*Da espandere a 800-1500 righe. Priorità: bassa (richiede studio preliminare 1-2 settimane).*
