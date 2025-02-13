import numpy as np

SQRT_2_DIV_PI = np.sqrt(2 / np.pi)


def calculate_indep_incr_square_data(data, start_interv_sec, delta):
    # avoid index out of range
    start_interv_sec = start_interv_sec[start_interv_sec + 3 * delta < data.shape[1]]
    data_start_intervals = data[:, start_interv_sec]

    inc0 = data[:, start_interv_sec + 3 * delta] - data_start_intervals
    inc1 = data[:, start_interv_sec + 2 * delta] - data_start_intervals
    inc2 = data[:, start_interv_sec + 1 * delta] - data_start_intervals
    inc0, inc1, inc2 = inc0.flatten(), inc1.flatten(), inc2.flatten()

    return inc0, inc1, inc2


def calculate_indep_incr_non_square_data(data, start_interv_sec, delta):
    # avoid index out of range
    start_interv_sec = start_interv_sec[start_interv_sec + 3 * delta < data.shape[1]]
    data_start_intervals = data[:, start_interv_sec]

    # only use data with at least 3*delta data points
    idx = np.sum(~np.isnan(data), axis=1) > 3 * delta
    d1_a, d0_a = data[idx], data_start_intervals[idx]

    inc0, inc1, inc2 = [], [], []
    for d1, d0 in zip(d1_a, d0_a):
        tmp0 = d1[start_interv_sec + 3 * delta] - d0
        tmp0 = tmp0[~np.isnan(tmp0)]
        inc0 += tmp0.tolist()

        tmp = d1[start_interv_sec + 2 * delta] - d0
        inc1 += tmp[: tmp0.size].tolist()

        tmp = d1[start_interv_sec + 1 * delta] - d0
        inc2 += tmp[: tmp0.size].tolist()

    inc0, inc1, inc2 = np.array(inc0), np.array(inc1), np.array(inc2)
    inc0, inc1, inc2 = inc0.flatten(), inc1.flatten(), inc2.flatten()

    return inc0, inc1, inc2


def wilcoxon_test(p1, p2):
    """
    Test if p(df)1 it is statistically distributed as p(df)2.
    """
    m = p1.size
    n = p2.size

    sp1 = np.sort(p1)
    sp2 = np.sort(p2)

    # Q = np.sum(sp2[:, np.newaxis] > sp1) is faster but memory inefficient
    Q = 0
    for val2 in sp2:
        Q += np.sum(val2 > sp1)

    Q_mean = n * m / 2
    Q_sigma = np.sqrt(n * m * (n + m + 1) / 12)
    T = np.abs(Q - Q_mean) / (Q_sigma * SQRT_2_DIV_PI)

    return T


def wilcoxon_test_multiple_bins(inc1, inc2, bins1, idx_c0):
    T_list = []
    for b1 in bins1:
        # - P(u_2|u_1) i.e. inc2 only where inc1 belongs to the nth bin
        idx_c1 = (inc1 > b1[0]) & (inc1 < b1[1])
        inc2_c1 = inc2[idx_c1]

        # - P(u_2|u_1,u_0=0)
        inc2_c1_c0 = inc2[idx_c1 & idx_c0]

        if inc2_c1.size > 40_000:  # we don't need that much data
            inc2_c1 = np.random.choice(inc2_c1, 40_000, replace=False)
        elif (
            inc2_c1.size < 30
        ):  # we need that much data (to be sure that we have enough data to calculate mean and have a Gaussian behaviour)
            continue  # skip this bin

        if inc2_c1_c0.size > 20_000:
            inc2_c1_c0 = np.random.choice(inc2_c1_c0, 20_000, replace=False)
        elif inc2_c1_c0.size < 30:
            continue

        # - test if P(u_2|u_1,u_0=0) is compatible with P(u_2|u_1)
        T = wilcoxon_test(inc2_c1, inc2_c1_c0)
        T_list.append(T)
    return np.mean(T_list)


def distribution(x_data, y_data, num_bin):
    # Compute the 2D histogram (counts) using the defined bins
    counts, bin_x_edges, bin_y_edges = np.histogram2d(x_data, y_data, bins=num_bin)
    counts = counts.astype(int)

    bin_x_width = bin_x_edges[1] - bin_x_edges[0]
    bin_y_width = bin_y_edges[1] - bin_y_edges[0]

    # Compute the joint probability density P(AnB)
    P_XnY = counts / (x_data.size * bin_x_width * bin_y_width)

    # Compute the marginal probabilities P(A) and P(B)
    P_X = (
        np.sum(P_XnY, axis=1) * bin_y_width
    )  # Multiply by bin_x_width since integration over y_data
    P_Y = (
        np.sum(P_XnY, axis=0) * bin_x_width
    )  # Multiply by bin_y_width since integration over x_data

    # Avoid division by zero by adding a small epsilon where necessary
    epsilon = np.finfo(float).eps
    P_X_nonzero = P_X + (P_X == 0) * epsilon
    P_Y_nonzero = P_Y + (P_Y == 0) * epsilon

    # Compute the conditional probabilities P(A|B) and P(B|A)
    P_XIY = P_XnY / P_Y_nonzero[np.newaxis, :]  # Broadcast over rows
    P_YIX = P_XnY / P_X_nonzero[:, np.newaxis]  # Broadcast over columns

    # Sum over rows and columns to get histograms for x and y
    counts_x = np.sum(counts, axis=1)
    counts_y = np.sum(counts, axis=0)

    return (
        P_XIY,
        P_YIX,
        P_XnY,
        P_X,
        P_Y,
        bin_x_edges,
        bin_y_edges,
        bin_x_width,
        bin_y_width,
        counts_x,
        counts_y,
    )


def compute_mean_values_per_bin(data, counts, bin_edges):
    num_bin = bin_edges.size - 1

    indices = np.digitize(data, bin_edges) - 1  # Convert to 0-based index
    indices = np.clip(indices, 0, num_bin - 1)
    sums = np.bincount(indices, weights=data, minlength=num_bin)

    mean_per_bin = np.full(sums.shape, np.nan)
    idx = counts != 0
    mean_per_bin[idx] = sums[idx] / counts[idx]

    return mean_per_bin
