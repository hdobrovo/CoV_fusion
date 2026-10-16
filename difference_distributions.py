import numpy as np

FILES = {
    "mers": "FINAL_boot_mers.npy",
    "mj": "FINAL_boot_mj.npy",
    "sars2": "FINAL_boot_sars2.npy",
}

PARAMETERS = ("gamma", "k", "D0")

N_DIFFERENCE_DRAWS = 100_000
RANDOM_SEED = 20261009


def load_bootstrap_parameters(path, parameters=PARAMETERS):
    """
    Return bootstrap distributions as:
    {'gamma': 1-D array, 'k': 1-D array, 'D0': 1-D array}.

    Accepted .npy formats:
      1. Structured array with fields gamma, k, D0.
      2. Dictionary saved using np.save(), with gamma, k, D0 keys.
      3. Two-dimensional numeric array with columns:
         [gamma, k, D0].
    """
    data = np.load(path, allow_pickle=True)
    # Dictionary saved with np.save(filename, dictionary)
    if data.shape == () and data.dtype == object:
        obj = data.item()

        if isinstance(obj, dict):
            missing = [p for p in parameters if p not in obj]

            if missing:
                raise KeyError(f"{path} is missing keys: {missing}")

            return {
                p: np.asarray(obj[p], dtype=float).ravel()
                for p in parameters
            }

    # Structured NumPy array
    if data.dtype.names is not None:
        missing = [p for p in parameters if p not in data.dtype.names]

        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")

        return {
            p: np.asarray(data[p], dtype=float).ravel()
            for p in parameters
        }

    # Standard numeric array, columns ordered gamma, k, D0
    data = np.asarray(data, dtype=float)

    if data.ndim != 2 or data.shape[1] < len(parameters):
        raise ValueError(
            f"{path} must be a structured array, a saved dictionary, "
            f"or a 2-D numeric array with columns {parameters}. "
            f"Received shape {data.shape}."
        )

    return {
        p: data[:, column].ravel()
        for column, p in enumerate(parameters)
    }


def independent_difference_distribution(
    values_a,
    values_b,
    n_draws=N_DIFFERENCE_DRAWS,
    rng=None,
):
    """
    Generate a bootstrap difference distribution for independent viruses.

    Each draw independently selects one estimate from virus A and one
    estimate from virus B, then computes A - B.
    """
    if rng is None:
        rng = np.random.default_rng()

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    if len(values_a) == 0 or len(values_b) == 0:
        raise ValueError("Bootstrap arrays cannot be empty.")

    indices_a = rng.integers(0, len(values_a), size=n_draws)
    indices_b = rng.integers(0, len(values_b), size=n_draws)

    return values_a[indices_a] - values_b[indices_b]


def percentile_ci(values, confidence_level=0.95):
    """
    Calculate a percentile bootstrap confidence interval.
    """
    alpha = 1.0 - confidence_level

    return np.quantile(
        values,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )


# Load all bootstrap parameter distributions.
boot = {
    virus: load_bootstrap_parameters(path)
    for virus, path in FILES.items()
}

# Define each contrast as first virus minus second virus.
comparisons = (
    ("mers", "sars2"),
    ("mers", "mj"),
    ("mj", "sars2"),
)

rng = np.random.default_rng(RANDOM_SEED)

print("Independent-bootstrap difference distributions")
print("95% percentile confidence intervals")
print("Difference = first virus minus second virus\n")

for virus_a, virus_b in comparisons:
    print(f"{virus_a.upper()} - {virus_b.upper()}")

    for parameter in PARAMETERS:
        difference_distribution = independent_difference_distribution(
            boot[virus_a][parameter],
            boot[virus_b][parameter],
            rng=rng,
        )

        median_difference = np.median(difference_distribution)
        ci_lower, ci_upper = percentile_ci(difference_distribution)

        print(
            f"  {parameter}: "
            f"median difference = {median_difference:.6g}; "
            f"95% CI = [{ci_lower:.6g}, {ci_upper:.6g}]"
        )

    print()
