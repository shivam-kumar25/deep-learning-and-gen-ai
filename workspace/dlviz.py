"""Small drawing helpers for the notes' figures (imported by figure cells; the live kernel runs in this folder too)."""
import numpy as np
import matplotlib.pyplot as plt


def draw_network(layers, ax=None, labels=None, layer_names=None, edge_labels=None, node_text=None, colors=None, title=None):
    """Draw a fully connected feedforward network.
    layers: units per layer, e.g. [2, 2, 1].  labels: per-layer list of node labels (or None).
    edge_labels: {(layer, i, j): text} for the edge from node i of `layer` to node j of `layer + 1`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1.9 * len(layers) + 1, 0.75 * max(layers) + 1.6))
    colors = colors or ["#dbeafe"] + ["#ede9fe"] * (len(layers) - 2) + ["#dcfce7"]
    pos = []
    for li, n in enumerate(layers):
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) if n > 1 else np.array([0.0])
        pos.append([(li * 2.2, -y) for y in ys])
    for li in range(len(layers) - 1):
        for i, (x0, y0) in enumerate(pos[li]):
            for j, (x1, y1) in enumerate(pos[li + 1]):
                ax.plot([x0, x1], [y0, y1], color="#9ca3af", lw=0.9, zorder=1)
                if edge_labels and (li, i, j) in edge_labels:
                    t = 0.3
                    ax.text(x0 + t * (x1 - x0), y0 + t * (y1 - y0) + 0.08, edge_labels[(li, i, j)], fontsize=8, ha="center", color="#374151",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=3)
    for li, layer in enumerate(pos):
        for i, (x, y) in enumerate(layer):
            ax.add_patch(plt.Circle((x, y), 0.32, fc=colors[li], ec="#374151", lw=1.2, zorder=2))
            txt = labels[li][i] if labels and labels[li] else (node_text(li, i) if node_text else "")
            if txt:
                ax.text(x, y, txt, ha="center", va="center", fontsize=9, zorder=4)
        if layer_names:   # all names on one line above the tallest layer
            ax.text(layer[0][0], max(max(p[1] for p in L) for L in pos) + 0.6, layer_names[li], ha="center", va="bottom", fontsize=9.5, color="#111827")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.8, 2.2 * (len(layers) - 1) + 0.8)
    top = max(max(p[1] for p in layer) for layer in pos)
    ax.set_ylim(-top - 0.7, top + (1.6 if layer_names else 0.7))
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def boundary(ax, predict, xlim=(-0.4, 1.4), ylim=(-0.4, 1.4), n=300, cmap="coolwarm", alpha=0.25):
    """Shade the regions of a 2-D classifier: predict(points [N, 2]) -> labels."""
    xx, yy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    z = np.asarray(predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=alpha, cmap=cmap)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
