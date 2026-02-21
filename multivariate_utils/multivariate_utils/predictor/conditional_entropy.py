import numpy as np
from collections import Counter
# from .down_sampling import HistogramDownsampler


def bin_data(X, edges):
    """Bin data using np.linspace edges."""
    return np.digitize(X, edges, right=True) - 1


def joint_prob_two(X, Y, edges):
    """
    Compute joint probability matrix P(X, Y) using binned data.
    Returns a 2D numpy array of shape (num_bins, num_bins).
    """
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    num_bins = len(edges) - 1
    total_samples = len(X)
    joint_matrix = np.zeros((num_bins, num_bins))

    for xb, yb in zip(X_bins, Y_bins):

        # check if indices are within bounds
        if 0 <= xb < num_bins and 0 <= yb < num_bins:

            # Increment the joint count
            joint_matrix[xb, yb] += 1

        # Normalize to get probabilities
    joint_matrix /= total_samples
    return joint_matrix


def joint_prob_three(X, Y, Z, edges):
    """
    Compute joint probability tensor P(X, Y, Z) using binned data.
    Returns a 3D numpy array of shape (num_bins, num_bins, num_bins).
    """
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    Z_bins = bin_data(Z, edges)
    num_bins = len(edges) - 1
    total_samples = len(X)
    joint_tensor = np.zeros((num_bins, num_bins, num_bins))

    for xb, yb, zb in zip(X_bins, Y_bins, Z_bins):

        # check if indices are within bounds
        if 0 <= xb < num_bins and 0 <= yb < num_bins and 0 <= zb < num_bins:

            # Increment the joint count
            joint_tensor[xb, yb, zb] += 1

        # Normalize to get probabilities
    joint_tensor /= total_samples
    return joint_tensor


def joint_prob_hist(X, Y, Z, edges):
    data = np.column_stack([X, Y, Z])
    hist, _ = np.histogramdd(data, bins=[edges, edges, edges])
    return hist / hist.sum()


def mi_no_hist_count(X, Y, edges):
    """
    NO histogramdd - pure counting
    YOUR EQUATION EXACTLY: ∑ p(x,y) log₂[p(x)p(y)/p(x,y)]
    """
    # Bin data
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    total_samples = len(X)

    # Count ALL joint events (like your 3D histogram)
    joint_counts = Counter(zip(X_bins, Y_bins))
    # I want  to get joint enrtropy

    I_neg = 0.0
    for (i, j), count_xy in joint_counts.items():
        p_xy = count_xy / total_samples

        # Compute marginals by summing over observed data (like your sum(axis))
        count_x = sum(c for (xb, yb), c in joint_counts.items() if xb == i)
        count_y = sum(c for (xb, yb), c in joint_counts.items() if yb == j)

        p_x = count_x / total_samples
        p_y = count_y / total_samples

        # YOUR EXACT EQUATION
        if p_xy > 0 and p_x > 0 and p_y > 0:
            I_neg += p_xy * np.log2((p_x * p_y) / p_xy)

    return -I_neg


def cmi_hist_for_loop(X, Y, Z, edges):
    p_xyz = joint_prob_three(X, Y, Z, edges)

    p_xz = p_xyz.sum(axis=1)
    p_yz = p_xyz.sum(axis=0)
    p_z = p_xyz.sum(axis=(0, 1))

    I = 0.0
    for i in range(p_xyz.shape[0]):
        for j in range(p_xyz.shape[1]):
            for k in range(p_xyz.shape[2]):
                p = p_xyz[i, j, k]
                if p > 0:

                    #  I(X;Y/Z) =

                    I += p * np.log2((p * p_z[k]) / (p_xz[i, k] * p_yz[j, k]))
    return I


def cmi_hist_numpy(X, Y, Z, edges):
    """
    Histogram-based conditional mutual information I(X;Y|Z)
    using fixed bin edges (np.linspace) and log2.
    """

    p_xyz = joint_prob_hist(X, Y, Z, edges)

    # Marginals
    p_xz = p_xyz.sum(axis=1, keepdims=True)  # (X,1,Z)
    p_yz = p_xyz.sum(axis=0, keepdims=True)  # (1,Y,Z)
    p_z = p_xyz.sum(axis=(0, 1), keepdims=True)  # (1,1,Z)

    mask = p_xyz > 0

    cmi = np.sum(
        p_xyz[mask] * np.log2((p_xyz[mask] * p_z[mask]) / (p_xz[mask] * p_yz[mask]))
    )

    return cmi


def mi_joint(X, Y, Z, edges):
    """
    Compute I(X; Y,Z) = ∑ p(x,y,z) log2 [ p(x,y,z) / ( p(x) p(y,z) ) ]

    Parameters
    ----------
    X, Y, Z : 1D array-like
        Input variables (already aligned sample-wise).
    edges : 1D array-like
        Bin edges (e.g. from np.linspace or np.quantile).

    Returns
    -------
    float
        I(X; Y,Z) in bits (non‑negative, up to discretization error).
    """
    # 1. Bin all variables
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    Z_bins = bin_data(Z, edges)
    total = len(X)

    # 2. Joint counts for (X,Y,Z)
    joint_counts = Counter(zip(X_bins, Y_bins, Z_bins))

    # 3. Marginal counts for X and (Y,Z)
    count_x = Counter(X_bins)
    count_yz = Counter(zip(Y_bins, Z_bins))

    # 4. Mutual information
    I = 0.0
    for (i, j, k), count_xyz in joint_counts.items():
        p_xyz = count_xyz / total
        p_x = count_x[i] / total
        p_yz = count_yz[(j, k)] / total

        if p_xyz > 0 and p_x > 0 and p_yz > 0:
            I += p_xyz * np.log2(p_xyz / (p_x * p_yz))

    return I


def mi_2d(X, Y, edges):
    """
    I(X;Y) = ∑ p(x,y) log[p(x,y)/(p(x)p(y))]
    Pure 2D — very stable
    """
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    total = len(X)

    joint_xy = Counter(zip(X_bins, Y_bins))
    cx = Counter(X_bins)
    cy = Counter(Y_bins)

    I = 0.0
    for (i, j), count_xy in joint_xy.items():
        p_xy = count_xy / total
        p_x = cx[i] / total
        p_y = cy[j] / total
        if p_xy > 0:
            I += p_xy * np.log2(p_xy / (p_x * p_y))
    return I


def cmi_as_2d_weighted_sum(X, Z, Y, edges):
    """
    I(X;Z|Y) = ∑_y p(y) × I(X;Z|Y=y)

    For EACH Y value, compute 2D MI between X and Z
    *** NO 3D SUMMATION EVER CREATED ***
    """
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    Z_bins = bin_data(Z, edges)
    total = len(X)

    count_y = Counter(Y_bins)
    cmi = 0.0

    # For each Y bin (only ~10 bins!)
    for y_val, count_y_val in count_y.items():
        if count_y_val == 0:
            continue

        p_y = count_y_val / total

        # Get data ONLY for this Y value
        mask = Y_bins == y_val
        X_slice = X_bins[mask]
        Z_slice = Z_bins[mask]

        # Compute 2D MI within this Y slice
        joint_xz = Counter(zip(X_slice, Z_slice))
        cx = Counter(X_slice)
        cz = Counter(Z_slice)

        i_xz_given_y = 0.0
        for (i, k), count_xz in joint_xz.items():
            p_xz = count_xz / count_y_val
            p_x = cx[i] / count_y_val
            p_z = cz[k] / count_y_val
            if p_xz > 0:
                i_xz_given_y += p_xz * np.log2(p_xz / (p_x * p_z))

        # Weight by p(y)
        cmi += p_y * i_xz_given_y

    return cmi


def mi_joint_NO_3D(X, Y, Z, edges):
    """
    I(X;Y,Z) = I(X;Y) + I(X;Z|Y)

    *** COMPLETELY AVOIDS 3D SUMMATION ***
    Uses chain rule to decompose into two 2D operations!
    """
    term1 = mi_2d(X, Y, edges)  # 2D: 100 bins
    term2 = cmi_as_2d_weighted_sum(X, Z, Y, edges)  # 10 × 2D: very dense

    return term1 + term2


def mi_x_yz(X, Y, Z, edges):
    """
    Compute I(X;Y,Z) by treating (Y,Z) as a single composite variable W = (Y,Z).

    I(X;Y,Z) = I(X;W)
             = ∑_{x,y,z} p(x,y,z) log2[ p(x,y,z) / ( p(x) p(y,z) ) ]

    Parameters
    ----------
    X, Y, Z : 1D arrays, shape (n_samples,)
        Input variables.
    edges : 1D array
        Bin edges (e.g. from np.linspace or np.quantile).

    Returns
    -------
    float
        I(X;Y,Z) in bits.
    """
    # 1. Bin each variable
    X_bins = bin_data(X, edges)
    Y_bins = bin_data(Y, edges)
    Z_bins = bin_data(Z, edges)
    total = len(X)

    # 2. Build composite variable W = (Y,Z) as pairs of bin indices
    W_bins = list(zip(Y_bins, Z_bins))  # each element is (y_bin, z_bin)

    # 3. Joint counts p(x,w) ⇒ p(x,y,z)
    joint_xw = Counter(zip(X_bins, W_bins))

    # 4. Marginals p(x) and p(w) ⇒ p(x) and p(y,z)
    count_x = Counter(X_bins)
    count_w = Counter(W_bins)

    # 5. MI computation: ∑ p(x,w) log2[p(x,w)/(p(x)p(w))]
    I = 0.0
    for (x_bin, w_bin), count_xw in joint_xw.items():
        p_xw = count_xw / total
        p_x = count_x[x_bin] / total
        p_w = count_w[w_bin] / total  # this is p(y,z)

        if p_xw > 0 and p_x > 0 and p_w > 0:
            I += p_xw * np.log2(p_xw / (p_x * p_w))

    return I


# def mi_x_yz_with_down_smaple(X, Y, Z, start_bin, down_smapl_bin, bin_range: tuple):
#     """
#     Compute I(X;Y,Z) by treating (Y,Z) as a single composite variable W = (Y,Z).

#     I(X;Y,Z) = I(X;W)
#              = ∑_{x,y,z} p(x,y,z) log2[ p(x,y,z) / ( p(x) p(y,z) ) ]

#     Parameters
#     ----------
#     X, Y, Z : 1D arrays, shape (n_samples,)
#         Input variables.
#     edges : 1D array
#         Bin edges (e.g. from np.linspace or np.quantile).

#     Returns
#     -------
#     float
#         I(X;Y,Z) in bits.
#     """
#     # 1. Bin each variable
#     # intiating downsampler
#     down_smapler = HistogramDownsampler(n_bins=start_bin)

#     histogram_3d = down_smapler.histogram_3d(x=X, y=Y, z=Z, bin_range=bin_range)
#     down_hist_3d = down_smapler.downsample_3d(
#         histogram=histogram_3d, target_bins=down_smapl_bin
#     )

#     count_3d = down_hist_3d["counts"]
#     total = down_hist_3d["total_counts"]
#     pxyz = count_3d / total
#     px = count_3d.sum(axis=(1, 2)) / total
#     pyz = count_3d.sum(axis=0) / total

#     # 5. MI computation: ∑ p(x,w) log2[p(x,w)/(p(x)p(w))]
#     # Then loop to compute MI
#     I = 0.0
#     for i in range(count_3d.shape[0]):  # x bins
#         for j in range(count_3d.shape[1]):  # y bins
#             for k in range(count_3d.shape[2]):  # z bins
#                 if pxyz[i, j, k] > 0 and px[i] > 0 and pyz[j, k] > 0:
#                     I += pxyz[i, j, k] * np.log2(pxyz[i, j, k] / (px[i] * pyz[j, k]))

#     return I
