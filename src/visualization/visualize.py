import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np

from src.models.buffered_cv import BufferedBlockedSplit


def plot_cv_indices(
    cv: BufferedBlockedSplit, X, ax, lw=10, print_stats=False,
):
    """Create a sample plot for indices of a BufferedBlockedSplit cross-validation object."""
    # make block labels
    blocks = np.full(len(X), cv._n_blocks - 1)
    block_size = len(X) // cv._n_blocks
    for i in range(cv._n_blocks - 1):
        blocks[i * block_size : (i + 1) * block_size] = i

    labels = ("Train", "Test", "Buffer")
    colors = ("tab:blue", "tab:orange", "lightgray")

    legend_lines = [  # not visible, just a hack for custom legend
        Line2D([0], [0], color=colors[0], lw=lw * 0.7),
        Line2D([0], [0], color=colors[1], lw=lw * 0.7),
        Line2D([0], [0], color=colors[2], lw=lw * 0.7),
    ]
    split_stats = {}

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.full(len(X), 2, dtype=np.uint8)
        indices[tr] = 0
        indices[tt] = 1
        split_stats[ii] = {
            "train": len(tr) / len(X),
            "test": len(tt) / len(X),
            "buffer": (len(X) - len(tr) - len(tt)) / len(X),
        }

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            np.full(len(indices), ii + 0.5),
            c=indices,
            marker="_",
            lw=lw,
            cmap=ListedColormap(colors),
        )

    # Plot the blocks at the end
    ax.scatter(
        range(len(X)),
        np.full(len(indices), ii + 1.5),
        c=blocks,
        marker="_",
        lw=lw,
        cmap=plt.cm.Paired,
    )

    # Formatting
    yticklabels = [f"Split {i}" for i in range(cv.n_splits)] + ["Blocks"]  # type: ignore
    ax.set(
        yticks=np.arange(cv.n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylim=[cv.n_splits + 1.2, -0.2],
        xlim=[0, len(X) - 1],
    )

    ax.legend(legend_lines, labels, bbox_to_anchor=(0.5, 1.1), loc="upper center", ncol=len(labels))

    if print_stats:
        print(*[f"Split {k}: {v}" for k, v in split_stats.items()], sep="\n")
    return ax
