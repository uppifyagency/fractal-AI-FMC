"""Tiny RGB rasterizer for RocketHookEnv.

Returns a 64x64x3 uint8 image — top-down view of the arena with the rocket,
hook, stone, target circle, and rope. No external deps beyond numpy.
This is what fragile.FractalTree streams into the dashboard for the best
walker (see fragile/app/_plangym.py PlanGymDisplay).
"""

from __future__ import annotations

import numpy as np


IMG_H = 64
IMG_W = 64

C_BG     = np.array([10, 12, 24], dtype=np.uint8)
C_GROUND = np.array([45, 35, 20], dtype=np.uint8)
C_ROCKET = np.array([240, 80, 80], dtype=np.uint8)
C_HOOK   = np.array([200, 200, 200], dtype=np.uint8)
C_ROPE   = np.array([130, 130, 130], dtype=np.uint8)
C_STONE_FALL = np.array([180, 180, 60], dtype=np.uint8)
C_STONE_HELD = np.array([60, 200, 220], dtype=np.uint8)
C_STONE_DONE = np.array([60, 220, 60], dtype=np.uint8)
C_TARGET = np.array([60, 220, 60], dtype=np.uint8)


def _world_to_pix(x: float, y: float, ax: tuple[float, float], ay: tuple[float, float]) -> tuple[int, int]:
    fx = (x - ax[0]) / (ax[1] - ax[0])
    fy = (y - ay[0]) / (ay[1] - ay[0])
    px = int(np.clip(fx * (IMG_W - 1), 0, IMG_W - 1))
    py = int(np.clip((1.0 - fy) * (IMG_H - 1), 0, IMG_H - 1))  # invert y for screen coords
    return px, py


def _draw_disc(img: np.ndarray, cx: int, cy: int, r: int, color: np.ndarray) -> None:
    h, w = img.shape[:2]
    y0 = max(cy - r, 0)
    y1 = min(cy + r + 1, h)
    x0 = max(cx - r, 0)
    x1 = min(cx + r + 1, w)
    for j in range(y0, y1):
        for i in range(x0, x1):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r * r:
                img[j, i] = color


def _draw_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: np.ndarray) -> None:
    # Bresenham
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    h, w = img.shape[:2]
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def render_rocket_hook(
    state: np.ndarray,
    target_xy: np.ndarray,
    arena_x: tuple[float, float],
    arena_y: tuple[float, float],
) -> np.ndarray:
    """Render the env state to a 64x64x3 uint8 RGB image."""
    img = np.tile(C_BG, (IMG_H, IMG_W, 1))
    # Ground band along the bottom
    ground_py = _world_to_pix(0.0, 0.0, arena_x, arena_y)[1]
    img[ground_py:, :] = C_GROUND

    rx, ry = state[0], state[1]
    hx, hy = state[6], state[7]
    sx, sy = state[10], state[11]
    phase = int(round(state[14]))

    # Target circle (3 px ≈ TARGET_RADIUS at this scale)
    tx, ty = _world_to_pix(float(target_xy[0]), float(target_xy[1]), arena_x, arena_y)
    _draw_disc(img, tx, ty, 3, C_TARGET // 3)
    # Rope rocket→hook
    p_rocket = _world_to_pix(rx, ry, arena_x, arena_y)
    p_hook = _world_to_pix(hx, hy, arena_x, arena_y)
    _draw_line(img, p_rocket, p_hook, C_ROPE)
    # Rocket
    _draw_disc(img, p_rocket[0], p_rocket[1], 2, C_ROCKET)
    # Hook
    _draw_disc(img, p_hook[0], p_hook[1], 1, C_HOOK)
    # Stone
    color = (C_STONE_FALL, C_STONE_HELD, C_STONE_DONE)[max(0, min(2, phase))]
    p_stone = _world_to_pix(sx, sy, arena_x, arena_y)
    _draw_disc(img, p_stone[0], p_stone[1], 2, color)

    return img
