import numpy as np


def calc_incs_1tau(v, tau):
    """v.shape = (n, time)"""

    tsize = v.shape[1]
    inc = v[:, tau:tsize] - v[:, 0 : tsize - tau]
    return inc.flatten()


def calc_incs_2tau(v, tau0, tau1):
    """
    v.shape = (n, time)
    tau0 > tau1
    """
    tsize = v.shape[1]

    d_01 = tau0 - tau1

    inc0 = v[:, tau0:tsize] - v[:, 0 : tsize - tau0]
    inc1 = v[:, tau1 : tsize - d_01] - v[:, 0 : tsize - tau0]

    inc0, inc1 = inc0.flatten(), inc1.flatten()

    return inc0, inc1


def calc_incs_3tau(v, tau0, tau1, tau2):
    """
    v.shape = (n, time)
    tau0 > tau1 > tau2
    """
    tsize = v.shape[1]

    d_01 = tau0 - tau1
    d_02 = tau0 - tau2

    inc0 = v[:, tau0:tsize] - v[:, 0 : tsize - tau0]
    inc1 = v[:, tau1 : tsize - d_01] - v[:, 0 : tsize - tau0]
    inc2 = v[:, tau2 : tsize - d_02] - v[:, 0 : tsize - tau0]

    inc0, inc1, inc2 = inc0.flatten(), inc1.flatten(), inc2.flatten()

    return inc0, inc1, inc2


def compute_indep_incs_square_data(data, start_interv_sec, delta):
    # avoid index out of range
    start_interv_sec = start_interv_sec[start_interv_sec + 3 * delta < data.shape[1]]
    data_start_intervals = data[:, start_interv_sec]

    inc0 = data[:, start_interv_sec + 3 * delta] - data_start_intervals
    inc1 = data[:, start_interv_sec + 2 * delta] - data_start_intervals
    inc2 = data[:, start_interv_sec + 1 * delta] - data_start_intervals
    inc0, inc1, inc2 = inc0.flatten(), inc1.flatten(), inc2.flatten()

    return inc0, inc1, inc2


def compute_indep_incs_non_square_data(data, start_interv_sec, delta):
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


def distribution(x_data, y_data, bins):
    # Compute the 2D histogram (counts) using the defined bins
    counts, bin_x_edges, bin_y_edges = np.histogram2d(x_data, y_data, bins=bins)
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
    bins = bin_edges.size - 1

    indices = np.digitize(data, bin_edges) - 1  # Convert to 0-based index
    indices = np.clip(indices, 0, bins - 1)
    sums = np.bincount(indices, weights=data, minlength=bins)

    mean_per_bin = np.full(sums.shape, np.nan)
    idx = counts != 0
    mean_per_bin[idx] = sums[idx] / counts[idx]

    return mean_per_bin


def get_Di_for_all_incs_and_scales(
    density_funcs_group,
    km_coeffs_group,
    bins: int,
    taylor_scale: float,
    use_Di_opti=True,
):
    """
    Creates the matrices:
       u_matrix, scale_matrix, D1_mat, D1_err_mat, D2_mat, D2_err_mat
    """
    n_scales = len(km_coeffs_group)

    # Preallocate
    u_matrix = np.full((n_scales, bins), np.nan)
    scale_matrix = np.full((n_scales, bins), np.nan)
    D1_mat = np.full((n_scales, bins), np.nan)
    D1_err_mat = np.full((n_scales, bins), np.nan)
    D2_mat = np.full((n_scales, bins), np.nan)
    D2_err_mat = np.full((n_scales, bins), np.nan)

    for i, (dens_funcs, km_est) in enumerate(zip(density_funcs_group, km_coeffs_group)):
        # valid_idxs = km_est.valid_idxs
        valid_idxs = km_est.valid_idxs

        count_valid = np.sum(valid_idxs)

        # fill U and scale
        u_matrix[i, :count_valid] = dens_funcs.mean_per_bin0[valid_idxs]
        scale_matrix[i, :count_valid] = km_est.scale / taylor_scale

        # choose which fields to extract (optimized vs non-optimized)
        if use_Di_opti:
            D1 = km_est.D1_opti
            D2 = km_est.D2_opti
        else:
            D1 = km_est.D1[valid_idxs]
            D2 = km_est.D2[valid_idxs]

        # fill D1, D2, and their errors
        D1_mat[i, :count_valid] = D1
        D1_err_mat[i, :count_valid] = km_est.D1_err[valid_idxs]
        D2_mat[i, :count_valid] = D2
        D2_err_mat[i, :count_valid] = km_est.D2_err[valid_idxs]

    return (
        u_matrix,
        scale_matrix,
        D1_mat,
        D1_err_mat,
        D2_mat,
        D2_err_mat,
    )
