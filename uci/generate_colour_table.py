import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib

from scipy.stats import norm
from typing import Dict, List, Tuple, Union

# INPUT
FILENAMES = ["benchmarks", "benchmarks_shifted"]
RUN_ON = "benchmarks"  # benchmarks_shifted
BASE_PATH = f"{pathlib.Path(__file__).parent.absolute()}"
UPPER = "↑"
LOWER = "↓"
N = 20  # N trials per dataset per method
# ----


def get_colors(num_methods: int) -> np.ndarray:
    colormap = plt.get_cmap("tab20b")
    colors = colormap(np.linspace(0, 1, num_methods))  # (num_methods)x4  (RGBA)
    return colors


def load_experiment(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert name in FILENAMES
    path = f"{BASE_PATH}/{name}.csv"
    path_std = f"{BASE_PATH}/{name}_std.csv"

    df = pd.read_csv(path, index_col=["experiment_name", "metric"])
    df_std = pd.read_csv(path_std, index_col=["experiment_name", "metric"])

    return df, df_std


def generate_img_filename(
    data_filename: str,
    metric: str,
) -> str:
    directory = f"{BASE_PATH}/performance_images/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return f"{directory}/winner_{data_filename}_{metric[:-1]}.png"


def generate_img(
    filename: str, metric: str, winner: np.ndarray, colors: np.ndarray
) -> str:
    num_datasets, num_methods = winner.shape  # (num_datasets)x(num_methods)
    rgb = np.zeros((num_methods, num_datasets, 3))  # (num_methods)x(num_datasets)x3
    for idx_method in range(num_methods):
        rgb[idx_method] = 1 - np.outer(winner[:, idx_method], colors[idx_method, :-1])
    outfn = generate_img_filename(filename, metric)
    mpl.image.imsave(outfn, rgb)
    return outfn


def print_table_header(metrics: List[str]) -> None:
    # Print table header
    print("\\begin{tabular}{|" + "c|" * (len(metrics) + 1) + "}")
    header = ""
    for metric in metrics:
        header += "  & \\textbf{" + metric[:-1].replace("_", "\\_") + "}"
    header += " \\\\ \\hline"
    print(header)


def print_table_footer() -> None:
    # Print table footer
    print("\\end{tabular}")


# =====
def test_mean_difference(
    dist_1: Dict[str, Union[float, int]],
    dist_2: Dict[str, Union[float, int]],
    level: float = 0.05,
) -> bool:
    """Returns if there is a statistical difference in the mean of dist_1 and dist_2"""
    n = norm(
        loc=0,
        scale=np.sqrt(
            (dist_1["std"] ** 2 / dist_1["n"]) + (dist_2["std"] ** 2 / dist_2["n"])
        ),
    )
    p_value = 2 * n.sf(abs(dist_1["mean"] - dist_2["mean"]))
    # Is there a statistical difference?
    return p_value < level


def get_winner(metric: str, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    if metric[-1] == LOWER:
        winner = means == np.nanmin(
            means, axis=1, keepdims=True
        )  # (num_datasets)x(num_methods)
    else:  # upper
        winner = means == np.nanmax(
            means, axis=1, keepdims=True
        )  # (num_datasets)x(num_methods)

    def test_draw(idx_ds: int, idx_winner: int, idx_contender: int) -> bool:
        dist_winner = {
            "mean": means[idx_ds, idx_winner],
            "std": stds[idx_ds, idx_winner],
            "n": N,
        }
        dist_contender = {
            "mean": means[idx_ds, idx_contender],
            "std": stds[idx_ds, idx_contender],
            "n": N,
        }
        return not test_mean_difference(dist_winner, dist_contender)

    # Check for statistical draws
    winner_with_draws = []
    winner_indices = np.where(
        winner
    )  # ([0, 1, ..., n_ds])x([idx_winner_ds_0, idx_winner_ds_1, ...])
    for idx_ds in winner_indices[0]:
        idx_winner_ds = winner_indices[1][idx_ds]
        winner_with_draws.append(
            [test_draw(idx_ds, idx_winner_ds, i) for i in range(winner.shape[1])]
        )
    winner_with_draws = np.array(winner_with_draws, dtype=bool)

    return winner_with_draws


# =====


if __name__ == "__main__":

    for i_fn, fn in enumerate(FILENAMES):
        # df = mean, df_std = standard deviation
        df, df_std = load_experiment(fn)

        methods = list(df.columns)
        num_methods = len(methods)
        metrics = list(df.index.get_level_values("metric").unique())

        colors = get_colors(num_methods)

        if i_fn == 0:
            print_table_header(metrics)

        # Analyse data
        line = fn.replace("_", "\\_")
        for metric in metrics:
            # extract data for given metric
            metric_mask = df.index.get_level_values("metric") == metric
            metric_mask_std = df_std.index.get_level_values("metric") == metric
            means = df[metric_mask].to_numpy()  # (num_datasets)x(num_methods)
            stds = df_std[metric_mask].to_numpy()  # (num_datasets)x(num_methods)

            # This is the code for my initial colorbar attempt
            # # Count number of first place wins for each method
            # if metric[-1] == lower:
            #     num_first_places = np.sum(data == data.min(axis=1, keepdims=True), axis=0)
            # else: # upper
            #     num_first_places = np.sum(data == data.max(axis=1, keepdims=True), axis=0)

            # # Produce bar plot
            # y = np.concatenate((np.zeros(1), num_first_places))
            # for k in range(len(methods)):
            #     plt.bar(y[k], 1, y[k+1], align='edge', color=colors[k])
            # plt.axis('off')
            # plt.axis('equal')
            # outfn = 'bar_' + fn + '_' + metric[:-1] + '.png'
            # plt.savefig(outfn)
            # plt.close()

            # Determine the winner
            winner = get_winner(metric, means, stds)  # (num_datasets)x(num_methods)

            # Produce little RGB winner plot
            outfn = generate_img(fn, metric, winner, colors)

            # Print table
            line += " & \\includegraphics[width=8cm]{" + outfn.split("/")[-1] + "}"
        print(line + "\\\\ \\hline")

    print_table_footer()
