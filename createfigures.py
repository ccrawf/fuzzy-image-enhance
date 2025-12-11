import matplotlib.pyplot as plt
import pandas as pd

nrmse_results = {
    "AGCWD": [0.272, 0.185, 1.118, 0.334, 0.064],
    "CLAHE": [0.127, 0.130, 0.722, 0.144, 0.095],
    "Fuzzy": [0.110, 0.164, 0.811, 0.225, 0.154],
    "HE": [0.398, 0.276, 2.851, 0.593, 0.431]
}

entropy_results = {
    "AGCWD": [0.523, -0.442, 0.991, 0.221, -0.619],
    "CLAHE": [0.361, 0.322, 0.784, 0.611, 0.337],
    "Fuzzy": [0.043, -0.139, 0.657, 0.459, 0.430],
    "HE": [-0.162, -0.069, -0.204, -0.281, -0.223]
}

tenengrad_results = {
    "AGCWD": [4.817, 4.078, 45.257, 6.084, -3.786],
    "CLAHE": [14.031, 23.577, 45.077, 10.988, 28.113],
    "Fuzzy": [9.416, 13.954, 32.037, 7.404, 20.450],
    "HE": [20.725, 17.263, 69.073, 11.678, 37.031]
}

metric_names = ["NRMSE", "Shannon Entropy", "Tenengrad Score"]
metrics = [nrmse_results, entropy_results, tenengrad_results]

for name, dict in zip(metric_names, metrics):
    # Create table
    df = pd.DataFrame(dict, index=[1,2,3,4,5])
    df.index.name = "Image #"

    print(f"{name} Results")
    print(df, "\n")

    plt.figure(figsize=(8,5))
    for method, values in dict.items():
        plt.plot(range(1,6), values, marker='o', label=method)

    plt.title(f"{name} Results")
    plt.axhline(0, color='black', alpha=0.3, )
    plt.xlabel("Image #")
    plt.ylabel("Score")
    plt.xticks([1,2,3,4,5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout

    fixed_name = name.lower().replace(" ","")
    plt.savefig(f"images_metrics/results/{fixed_name}.png", dpi=300)