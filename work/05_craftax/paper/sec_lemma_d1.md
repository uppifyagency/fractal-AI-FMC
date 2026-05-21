# Lemma D.1 — Compounding monotonicity under regime separation

> Theoretical sketch backing Conjecture D (Gap 6). Sympy-verified
> derivations are in `gap6_lemma_d1_results.txt`. Workshop reviewers can
> accept this as informal; conference reviewers will want the full
> proof — extension below indicates which steps need tightening.

## Setup

Recall the FMC `relativize` map applied to walker cumulative rewards:

$$
\widehat{r} = \mathrm{relativize}(r) =
\begin{cases}
1 + \log\!\bigl(1 + z\bigr), & z > 0 \\[2pt]
\dfrac{e^{z}}{e}, & z \le 0
\end{cases}
\qquad z = \frac{r - \bar r}{\sigma_r}
$$

where $\bar r$, $\sigma_r$ are the population mean / std at the current tick.
The cloning probability of walker $i$ is

$$
P_i^{\text{clone}} = \frac{1}{N} \sum_{j \ne i}
\beta \cdot \max\!\left(0,\; 1 - \frac{\widehat{r}_i}{\widehat{r}_j}\right)
$$

so a walker with very large $\widehat{r}_i$ resists cloning, and a walker with
small $\widehat{r}_i$ adopts the trajectory of a high-$\widehat{r}$ peer.

**Setup for Lemma D.1.** Assume a single *firing walker* (one whose
trajectory triggered an unseen achievement, picking up $w_j \in [50, 300]$
of sparse bonus) so $r_{\text{firing}} \gg r_{\text{other}} \;\forall\, \text{other}$.
The firing walker's $z_{\text{firing}} \gg 1$ → it lives in the **log regime**.
The non-firing walkers live near $z \approx 0$ → **exponential regime**.

A tier amplification raises the inv-tier weight $\lambda_T \to \mu_T \cdot \lambda_T$
with $\mu_T \in (1, c)$, $c \approx 4$. The effect on cum_rewards:

- *firing walker*: reward dominated by $w_j$; the $\Delta r$ from inv-tier
  amplification is $O(\Delta\lambda_T \cdot |\text{inventory}|)$ which is
  small relative to $w_j$. Approximately $r_{\text{firing}}^{(k+1)} \approx
  r_{\text{firing}}^{(k)}$.
- *all walkers* (including non-firing): mean $\bar r$ rises by $\Delta\bar r > 0$;
  std rises by $\Delta\sigma$.

Empirically (verified across exp01–17): $\Delta\bar r \gg \Delta\sigma$
in the regime where shaping is helping rather than collapsing. Specifically
$\Delta\sigma / \Delta\bar r \in [0.05, 0.25]$ across the productive
amplification window (Falsifications 1, 2 mark the boundary at ≥1.4× per
step where this ratio inverts).

## Statement

**Lemma D.1.** Let $\Phi(R^{(k)})$ denote the Crafter score under reward
function $R^{(k)} = R_{\text{env}} + \alpha_{\text{inv}} R_{\text{inv}}^{(\boldsymbol\lambda \odot \boldsymbol\mu_{1:k})} + R_{\text{ach}}^{(\mathbf{w}_{\text{tier}})}$,
where $\boldsymbol\mu_{1:k}$ stacks $k$ tier amplifications. Under the regime
separation assumption above and the bound $\Delta\sigma / \Delta\bar r < z_{\text{firing}}^{-1}$,

$$
\boxed{\Phi\bigl(R^{(k+1)}\bigr) \;>\; \Phi\bigl(R^{(k)}\bigr) \;-\; \delta_n}
$$

with noise term $\delta_n = O(n^{-1/2})$ from finite-sample seed estimation.

## Proof sketch

**Step 1 — change in firing walker's $\widehat{r}$.** From the sympy-verified
Taylor expansion (`gap6_lemma_d1_results.txt`, R2'):

$$
\Delta \widehat{r}_{\text{firing}} = \frac{-\,\Delta\bar r \cdot \sigma\,(z+1)
\;+\; \Delta\sigma\,\bigl[\Delta\bar r - \sigma z(z+1)\bigr]}{\sigma^{2}(z+1)^{2}}
$$

Dropping second-order terms in $(\Delta\bar r,\Delta\sigma)$ and substituting
$z = z_{\text{firing}}$:

$$
\Delta \widehat{r}_{\text{firing}} \;\approx\;
-\,\frac{\Delta\bar r + z_{\text{firing}}\,\Delta\sigma}{\sigma_r\,(1 + z_{\text{firing}})}
$$

This is **negative**: the firing walker's relativized score *decreases* under
amplification. But the magnitude is bounded by $1/(1 + z_{\text{firing}})$ —
which is small when $z_{\text{firing}} \gg 1$.

**Numerical instantiation** (Craftax-like, exp17): $r_{\text{firing}} \approx 250$,
$\bar r \approx 80$, $\sigma_r \approx 35$ → $z_{\text{firing}} = 4.86$.
A typical tier-amp step gives $\Delta\bar r \approx 8$, $\Delta\sigma \approx 1.5$:

$$
\Delta \widehat{r}_{\text{firing}} \;\approx\;
-\,\frac{8 + 4.86 \cdot 1.5}{35 \cdot 5.86} \;\approx\; -0.074
$$

**Step 2 — change in non-firing walkers' $\widehat{r}$.** They sit in the
exp regime with $z \approx 0$. The first-order shift is

$$
\Delta \widehat{r}_{\text{other}} \;\approx\;
\frac{\Delta\bar r}{\sigma_r \cdot e}
\;=\; \frac{8}{35 \cdot 2.718} \;\approx\; +0.084
$$

(using $\widehat{r} \approx e^z / e \approx 1/e + z/e$ for small $z$).

**Step 3 — sign of the shift in cloning probability ratio.** The cloning
probability $P_i^{\text{clone}}$ depends on $\widehat{r}$-ratios. The firing
walker's *floor advantage* is

$$
\widehat{r}_{\text{firing}} - \widehat{r}_{\text{other}}
$$

After amplification:

$$
\widehat{r}_{\text{firing}}^{(k+1)} - \widehat{r}_{\text{other}}^{(k+1)}
\;\approx\;
\bigl(\widehat{r}_{\text{firing}} - 0.074\bigr) -
\bigl(\widehat{r}_{\text{other}} + 0.084\bigr)
\;=\; \widehat{r}_{\text{firing}} - \widehat{r}_{\text{other}} - 0.158
$$

The advantage **shrinks** by $0.158$. But the firing walker's $\widehat{r} =
2.77$, the non-firing walker's $\widehat{r} \approx 0.37$ → starting gap of
$2.40$. A 0.158 dent on a 2.40 gap is a 6.6 % erosion — well below the
threshold where cloning would *fail* to replicate the firing trajectory.

**Step 4 — replication count.** The expected number of walkers cloning the
firing trajectory in one tick is

$$
\mathbb{E}[N_{\text{clone}}] = (N - 1) \cdot \beta \cdot
\max\!\left(0,\; 1 - \frac{\widehat{r}_{\text{other}}}{\widehat{r}_{\text{firing}}}\right)
$$

Pre-amplification: $1 - 0.37/2.77 = 0.866$.
Post-amplification: $1 - 0.45/2.69 = 0.832$.

So **per tick** the replication count drops by 4 % at most. The firing
trajectory still gets ~83 % of the population in one step — fast enough to
keep the achievement-completing rollout dominant.

**Step 5 — chain compounding.** The argument applies independently to each
*new* achievement that fires during the rollout. As successive tiers
$T_1, T_2, \dots, T_L$ are added, each enables a different sub-chain to fire
without disrupting the prior fires. Since the firing-walker count for each
sub-chain remains $\Theta(N)$ in expectation, the *aggregate* unlock rate
$\rho_j$ for tiered achievements increases monotonically.

The Hafner geometric mean $\Phi$ amplifies log-scale increases in any
$\rho_j$, so $\Phi(R^{(k+1)}) > \Phi(R^{(k)})$ up to seed noise $\delta_n$.

$\blacksquare$ (sketch)

## Tightened version (v2): finite-sample bounds

Three sympy-verified extensions of the workshop sketch
(`gap6_lemma_d1_v2_results.txt`):

### Theorem T1 — sufficient condition for regime separation

Model walker cum-rewards as $r_i = \text{base}_i + \mathbb{1}_{\text{firing}_i} \cdot w_j$
where firing happens with probability $p$ per walker per rollout. The
population-wide firing-walker $z$-score is

$$
z_{\text{firing}}(p, w_j, \sigma_{\text{base}})
= \frac{(1 - p)\, w_j}{\sqrt{p(1-p)\,w_j^2 + \sigma_{\text{base}}^2}}.
$$

To guarantee $z_{\text{firing}} \ge z^* > 0$ (log-regime separation), the
bonus weight must satisfy

$$
w_j \;\ge\; w_{\min}(z^*, \sigma_{\text{base}}, p)
= \frac{\sigma_{\text{base}} \cdot z^*}{\sqrt{(1-p)(1 - p - z^{*2} p)}}
$$

valid when $p < 1/(1 + z^{*2})$.

**Numerical instantiation (Craftax exp17, make_iron_pickaxe):**
$p = 1/3$, $\sigma_{\text{base}} = 35$, $z^* = 1.0$:

$$
w_{\min} = 74.25 \quad < \quad 200 = w_{\text{exp17}}
$$

The exp17 weight has a **2.7× safety margin** over the regime-separation
threshold. At $z^* = 1.3$ the threshold rises to 173 — exp17 sits at
the edge of the high-separation regime, consistent with the observation
that further pushing $w_j$ on iron-tier produced diminishing returns
(exp23: $w \to 250$ gave $\Delta\Phi = -12$ pp).

### Theorem T2 — finite-sample bound on $\mathbb{E}[N_{\text{clone}}]$

Per FMC tick, each non-firing walker independently clones onto a
randomly-drawn partner with probability $1 - \rho$ where
$\rho = \widehat{r}_{\text{other}} / \widehat{r}_{\text{firing}}$.
The number of walkers cloning onto the firing trajectory in one tick:

$$
N_{\text{clone}} \sim \mathrm{Binomial}\bigl(N - 1, \,1 - \rho\bigr)
$$

with $\mathbb{E}[N_{\text{clone}}] = (N-1)(1-\rho)$ and a Hoeffding tail
bound:

$$
\mathbb{P}\bigl(\,|N_{\text{clone}} - \mathbb{E}[N_{\text{clone}}]| \ge t\,\bigr)
\;\le\; 2\exp\!\bigl(-2t^2 / (N-1)\bigr).
$$

For $N = 512$ and $\rho = 0.13$ (Craftax-like), $\mathbb{E}[N_{\text{clone}}] \approx 445$.
Setting the tail bound to 5 % gives $t = 30.7$:

$$
\mathbb{P}\bigl(N_{\text{clone}} \in [414, 475]\bigr) \ge 0.95
$$

The firing trajectory dominates with high probability:
$N_{\text{clone}} = \Theta(N)$ at every tick.

### Theorem T3 — end-to-end Φ monotonicity bound

A tier amplification step that lifts the per-rollout firing probability
$p_j$ for a tier-$j$ achievement by a multiplicative factor $(1 + \delta_p)$
produces a Crafter-score change

$$
\frac{\Delta \Phi}{\Phi} \;\approx\; \frac{1}{J} \cdot
\frac{100 \,\delta_p\, p_{j,\text{old}}}{1 + 100\, p_{j,\text{old}}}.
$$

For $p_{\text{old}} = 1/3$, $\delta_p = 0.2$, $J = 22$:
$\Delta\Phi/\Phi \approx 0.0088$ → at $\Phi \approx 50$, predicted gain
$\approx 0.44$ pp per single-achievement tier-amp step.

This is **conservative**: observed per-step gains in the additive
trajectory are 0.5–4.7 pp, an order-of-magnitude larger because each
amplification step affects *multiple* achievements simultaneously
(e.g. iron-tier inv amp benefits `make_iron_pickaxe`, `collect_iron`,
`collect_coal`, `make_iron_sword` in parallel). T3 is therefore a
**lower bound on monotonicity**, consistent with all observed
positive Δs.

## What still needs more work

1. **Regime separation under multi-firing.** When several achievements
   fire in the same rollout, the walker's $z$ depends on the joint
   distribution; T1 covers single-firing only.
2. **Coupling of cloning rounds.** T2 bounds one tick; an end-to-end
   bound on $M$ ticks requires martingale-style concatenation.
3. **Wright-Fisher closure.** The branching process underlying FMC
   walker dynamics maps to Wright-Fisher with mutation; closing the
   formal connection (cf. `work/02_deep_dives/07_wright_fisher_mapping.md`)
   gives a sharper $\Theta(\sqrt{N})$ vs $\Theta(N)$ analysis but is
   beyond the workshop scope.

Cross-benchmark replication (Gap 4) is the empirical bound on whether the
lemma holds task-independently. If Crafter-original shows the same
monotonic compounding pattern, the lemma is empirically validated even
without item 3 above.

## Quantitative falsification thresholds

The lemma's regime-separation assumption breaks when:

- $\sigma_r$ explodes from over-amplification (exp04, $\eta = 6.67$): firing
  walker's $z$ collapses from $\sim 5$ to $\sim 1$; the log-regime advantage
  vanishes; observed $\Delta\Phi = -4$ pp.
- $\alpha > 1$ in $r^\alpha$ (exp22): cum_rewards become arbitrarily peaked
  on the $\arg\max$, so even non-firing walkers transition into the log
  regime; the firing walker loses uniqueness; observed $\Delta\Phi = -24$ pp.

Both failures are predicted by the lemma's hypothesis (regime separation)
breaking, and the lemma correctly forecasts the qualitative collapse.
