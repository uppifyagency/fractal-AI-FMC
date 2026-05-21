# LLM-generated gridworld world-model — meta/llama-3.3-70b-instruct
# E1-LLM Route B (code form). 1 attempt(s).

def transition(r, c, done, action, grid, H, W):
    if done:
        return r, c, True

    nr, nc = r, c
    if action == 0 and r > 0:
        nr -= 1
    elif action == 1 and r < H - 1:
        nr += 1
    elif action == 2 and c > 0:
        nc -= 1
    elif action == 3 and c < W - 1:
        nc += 1

    ndone = grid[nr][nc] in [1, 2]
    return nr, nc, ndone
