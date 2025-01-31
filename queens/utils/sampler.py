import numpy as np


def _compute_covariance(samples, weights, mean):
    num_samples = len(samples)
    covariance = -np.outer(mean, mean)
    if weights is None:
        weights = np.ones(num_samples) / num_samples
    for weight, sample in zip(weights, samples):
        covariance += weight * np.outer(sample, sample)
    return covariance * len(samples) / (len(samples) - 1)  # unbiased


def get_samples_statistics(samples, weights=None, covariance=False, **kwargs):
    if weights is not None:
        weights = weights.reshape(-1)

    num_samples = samples.shape[0]
    dimension = samples.shape[-1]
    mean = np.average(samples, weights=weights, axis=0)
    squared_mean = np.average(samples**2, weights=weights, axis=0)
    variance = (squared_mean - mean**2) * len(samples) / (len(samples) - 1)  # unbiased

    description = {}
    description["dimension"] = dimension
    description["num_samples"] = num_samples
    description["mean"] = mean
    description["variance"] = variance
    description["standard_deviation"] = np.sqrt(variance)
    description["min"] = samples.min(axis=0)
    description["max"] = samples.max(axis=0)

    if covariance:
        if samples.ndim == 2:
            if samples.shape[1] != 1:
                covariance = _compute_covariance(samples, weights, mean)
                description["covariance"] = covariance
        elif samples.ndim == 3:
            covariance = np.array(
                [
                    _compute_covariance(samples[:, i, :], weights, mean[i])
                    for i in range(samples.shape[1])
                ]
            )
            description["covariance"] = covariance
        elif samples.ndim > 3:
            raise ValueError(f"Could not compute covariance for an {samples.shapes} array.")

    return description
