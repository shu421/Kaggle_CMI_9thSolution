import numpy as np

def cross_correlation(signal, pattern):
    """
    Compute cross correlation between signal and pattern
    """
    correlation = np.correlate(signal, pattern, mode="valid")
    normalized_correlation = correlation / (
        np.linalg.norm(pattern)
        * np.sqrt(np.convolve(signal**2, np.ones(len(pattern)), "valid"))
    )
    return normalized_correlation


def find_similar_steps(signal, window_size=17280, threshold=1.0):
    """
    Compare waveforms of each bin and find pairs with high similarity.
    """
    num_bins = len(signal) // window_size
    similar_steps = set()

    for i in range(num_bins):
        for j in range(i + 1, num_bins):
            bin_i = signal[i * window_size : (i + 1) * window_size]
            bin_j = signal[j * window_size : (j + 1) * window_size]
            correlation = max(cross_correlation(bin_i, bin_j))
            if correlation > threshold:
                similar_steps.add(i * window_size)
                similar_steps.add(j * window_size)

    return sorted(list(similar_steps))
