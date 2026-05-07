import numpy as np

def distribute_nodes_spatially(n_nodes, n_groups):
    rows = int(np.sqrt(n_nodes))
    cols = (n_nodes + rows - 1) // rows

    positions = []
    for x in range(cols):
        for y in range(rows):
            idx = x * rows + y
            if idx < n_nodes:
                positions.append((x, y))
    positions = np.array(positions)

    configs = []
    for gr_rows in range(1, n_groups + 1):
        gr_cols = (n_groups + gr_rows - 1) // gr_rows  # ceil
        total = gr_rows * gr_cols
        if total >= n_groups:
            configs.append((gr_rows, gr_cols, total))

    configs.sort(key=lambda x: (abs(x[0] - x[1]), x[2]))

    gr_rows, gr_cols, _ = configs[0]

    block_w = cols / gr_cols
    block_h = rows / gr_rows

    node_groups = {id: [] for id in range(n_groups)}
    for index, (x, y) in enumerate(positions):
        bx = min(int(x // block_w), gr_cols - 1)
        by = min(int(y // block_h), gr_rows - 1)
        group_id = by * gr_cols + bx
        if group_id >= n_groups:
            group_id = n_groups - 1  # fallback
        node_groups[group_id].append(index)

    return node_groups