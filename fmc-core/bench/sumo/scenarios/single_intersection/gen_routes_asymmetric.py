"""Asymmetric traffic generator — N/S heavy, E/W light.

This is where static-cycle controllers should fail (allocate equal time to a
direction that doesn't need it) and adaptive controllers should win.
"""

from __future__ import annotations

import argparse
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate-heavy", type=float, default=0.30, help="N/S arrival rate")
    p.add_argument("--rate-light", type=float, default=0.05, help="E/W arrival rate")
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="routes.rou.xml")
    args = p.parse_args()

    rng = random.Random(args.seed)

    origins = {
        "n_to_c": ("c_to_s", "c_to_w", "c_to_e", args.rate_heavy),
        "s_to_c": ("c_to_n", "c_to_e", "c_to_w", args.rate_heavy),
        "e_to_c": ("c_to_w", "c_to_n", "c_to_s", args.rate_light),
        "w_to_c": ("c_to_e", "c_to_s", "c_to_n", args.rate_light),
    }
    direction_probs = [0.6, 0.2, 0.2]

    vehicles = []
    veh_id = 0
    for o, (s, l, r, rate) in origins.items():
        dests = [s, l, r]
        t = 0.0
        while t < args.duration:
            inter = rng.expovariate(rate)
            t += inter
            if t >= args.duration:
                break
            d = rng.choices(dests, weights=direction_probs, k=1)[0]
            vehicles.append((t, veh_id, o, d))
            veh_id += 1

    vehicles.sort(key=lambda v: v[0])

    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    out.append('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>')
    for _, vid, o, d in vehicles:
        out.append(f'    <route id="r_{vid}" edges="{o} {d}"/>')
    for t, vid, _, _ in vehicles:
        out.append(f'    <vehicle id="v{vid}" type="car" route="r_{vid}" depart="{t:.2f}"/>')
    out.append("</routes>")

    with open(args.output, "w") as f:
        f.write("\n".join(out))
    n_heavy = sum(1 for _, _, o, _ in vehicles if o in ("n_to_c", "s_to_c"))
    n_light = sum(1 for _, _, o, _ in vehicles if o in ("e_to_c", "w_to_c"))
    print(f"Generated {len(vehicles)} vehicles ({n_heavy} N/S + {n_light} E/W) over {args.duration}s")
    print(f"Heavy:Light ratio = {n_heavy/max(1,n_light):.2f}")


if __name__ == "__main__":
    main()
