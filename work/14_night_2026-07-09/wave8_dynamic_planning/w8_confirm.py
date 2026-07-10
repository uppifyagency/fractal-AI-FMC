#!/usr/bin/env python3
"""Independent confirmation of the adversarial overturn: decode fix + tuning."""
import sys, os
import numpy as np
from math import sqrt, erfc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))
from w8_deceptive_nav import DeceptiveNav, run_episode, plan_random_shooting, plan_cem
from w8_adv_tune import make_fmc_planner   # reviewer's variant (bit-identical to core.plan on majority)

def z2(k1,n1,k2,n2):
    p1,p2=k1/n1,k2/n2; p=(k1+k2)/(n1+n2); se=sqrt(p*(1-p)*(1/n1+1/n2))
    return (0.0,1.0) if se==0 else ((p1-p2)/se, erfc(abs((p1-p2)/se)/sqrt(2)))

def ev(planner,N,M,off,n,H=70):
    s=0
    for inst in range(n):
        rng=np.random.default_rng(90210+inst*31+N*7)
        s+=int(run_episode(DeceptiveNav(offset=off,reward_mode="dense"),planner,N,M,H,rng)["success"])
    return s

for (N,M) in [(36,11),(48,12)]:
    n=40; off=1.5
    rs=ev(plan_random_shooting,N,M,off,n); cem=ev(plan_cem,N,M,off,n)
    bb=max(rs,cem); bn="rand" if rs>=cem else "CEM"
    configs={"FMC-maj(1,1)":make_fmc_planner(1.0,1.0,"majority"),
             "FMC-arg(1,1)":make_fmc_planner(1.0,1.0,"argmax"),
             "FMC-arg(.5,2)":make_fmc_planner(0.5,2.0,"argmax")}
    print(f"=== B={N*M} (N={N},M={M}) offset={off} n={n} ===")
    print(f"  rand-shoot={rs/n:.3f}  CEM={cem/n:.3f}  best_base={bb/n:.3f}({bn})")
    for name,pl in configs.items():
        k=ev(pl,N,M,off,n); z,p=z2(k,n,bb,n)
        print(f"  {name:14s}={k/n:.3f}  vs best_base z={z:+.2f} p={p:.3f}")
