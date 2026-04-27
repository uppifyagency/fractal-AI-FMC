"""Generate Poisson-arrival route file for the single-intersection scenario.

Usage:
    python gen_routes.py --rate 0.4 --duration 1800 --seed 42 > routes.rou.xml

The arrival rate is per-direction per-second. With rate=0.4 and 4 directions,
expected ~1.6 vehicles/second total = high but congestible. Duration 1800s = 30 min.
"""

from __future__ import annotations

import argparse
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=0.3)
    p.add_argument("--duration", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="routes.rou.xml")
    args = p.parse_args()

    rng = random.Random(args.seed)

    # 12 routes total (4 origins × 3 destinations each).
    # Through (straight) is most common; left and right less common.
    routes = []
    origins = {
        "n_to_c": ["c_to_s", "c_to_w", "c_to_e"],   # straight, left, right
        "s_to_c": ["c_to_n", "c_to_e", "c_to_w"],
        "e_to_c": ["c_to_w", "c_to_n", "c_to_s"],
        "w_to_c": ["c_to_e", "c_to_s", "c_to_n"],
    }
    # Probabilities for [straight, left, right].
    direction_probs = [0.6, 0.2, 0.2]

    vehicles = []
    veh_id = 0
    for o, dests in origins.items():
        # Independent Poisson arrivals per direction.
        t = 0.0
        while t < args.duration:
            inter = rng.expovariate(args.rate)
            t += inter
            if t >= args.duration:
                break
            # pick destination by direction probability
            d = rng.choices(dests, weights=direction_probs, k=1)[0]
            vehicles.append((t, veh_id, o, d))
            veh_id += 1

    vehicles.sort(key=lambda v: v[0])

    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    out.append('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>')
    # Define routes inline with each vehicle.
    for t, vid, o, d in vehicles:
        route_id = f"r_{vid}"
        out.append(f'    <route id="{route_id}" edges="{o} {d}"/>')
    for t, vid, o, d in vehicles:
        route_id = f"r_{vid}"
        out.append(f'    <vehicle id="v{vid}" type="car" route="{route_id}" depart="{t:.2f}"/>')
    out.append("</routes>")

    with open(args.output, "w") as f:
        f.write("\n".join(out))
    print(f"Generated {len(vehicles)} vehicles over {args.duration}s ({len(vehicles)/args.duration:.2f} veh/s) → {args.output}")


if __name__ == "__main__":
    main()
