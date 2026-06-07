import copy
from warnings import warn

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.lapack import dgesvd as svd

from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.testing.decorators import warning_for_keywords


def dimensionality_problem_message(arr, num_samples, spr):
    return (
        f"Number of samples {arr.shape[-1]} - 1 < Dimensionality "
        f"{num_samples}. This might have a performance impact. Increase p"
        f"atch_radius to {spr} to avoid this."
    )


def _pca_classifier_complex(L, nvoxels):
    """
    PCA classifier for complex data using Complex Wishart distribution
    """
    if L.size > nvoxels - 1:
        L = L[-(nvoxels - 1) :]
    
    # For complex Wishart, we need different scaling
    n_noise = max(1, int(0.20 * len(L)))
    noise_vals = np.sort(L)[:n_noise]
    var = np.mean(noise_vals)

    c = L.size - 1
    # Complex Wishart scaling factor is different
    r = L[c] - L[0] - 2 * np.sqrt(2 * (c + 1.0) / nvoxels) * var
    while r > 0:
        var = np.mean(L[:c])
        c = c - 1
        r = L[c] - L[0] - 2 * np.sqrt(2 * (c + 1.0) / nvoxels) * var
    ncomps = c + 1

    return var, ncomps


def create_patch_radius_arr(arr, pr):
    patch_radius = copy.deepcopy(pr)

    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius).astype(int)
    patch_radius[arr.shape[0:3] == np.ones(3)] = 0  # account for dim of size 1

    return patch_radius


def compute_patch_size(patch_radius):
    return 2 * patch_radius + 1


def compute_num_samples(patch_size):
    return np.prod(patch_size)


def compute_suggested_patch_radius(arr, patch_size):
    tmp = np.sum(patch_size == 1)  # count spatial dimensions with size 1
    if tmp == 0:
        root = np.ceil(arr.shape[-1] ** (1.0 / 3))  # 3D
    if tmp == 1:
        root = np.ceil(arr.shape[-1] ** (1.0 / 2))  # 2D
    if tmp == 2:
        root = arr.shape[-1]  # 1D
    root = root + 1 if (root % 2) == 0 else root  # make odd

    return int((root - 1) / 2)


@warning_for_keywords()
def genpca_complex(
    arr,
    *,
    sigma=None,
    mask=None,
    patch_radius=2,
    pca_method="eig",
    tau_factor=None,
    return_sigma=False,
    out_dtype=None,
    suppress_warning=False,
):
    """
    Complex MP-PCA denoising for complex-valued data
    
    Parameters
    ----------
    arr : ndarray
        4D complex array (x, y, z, measurements)
    sigma : float or ndarray, optional
        Noise standard deviation
    mask : ndarray, optional
        Boolean mask
    patch_radius : int or list
        Patch radius for local PCA
    pca_method : str
        'eig' or 'svd' (default: 'eig')
    tau_factor : float, optional
        Threshold factor
    return_sigma : bool
        Whether to return estimated sigma
    out_dtype : dtype
        Output data type
    suppress_warning : bool
        Whether to suppress warnings
    """
    
    # Ensure input is complex
    if not np.iscomplexobj(arr):
        raise ValueError("Input array must be complex-valued for Complex MP-PCA")
    
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)[..., 0]

    if out_dtype is None:
        out_dtype = arr.dtype

    # Set calculation dtype based on input complexity
    if arr.dtype == np.complex128:
        calc_dtype = np.complex128
        real_dtype = np.float64
    else:
        calc_dtype = np.complex64
        real_dtype = np.float32

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.", arr.shape)

    if pca_method.lower() == "svd":
        is_svd = True
    elif pca_method.lower() == "eig":
        is_svd = False
    else:
        raise ValueError("pca_method should be either 'eig' or 'svd'")

    patch_radius_arr = create_patch_radius_arr(arr, patch_radius)
    patch_size = compute_patch_size(patch_radius_arr)

    ash = arr.shape[0:3]
    if np.any((ash != np.ones(3)) * (ash < patch_size)):
        raise ValueError("Array 'arr' is incorrect shape")

    num_samples = compute_num_samples(patch_size)
    if num_samples == 1:
        raise ValueError("Cannot have only 1 sample, please increase patch_radius.")
    
    if (num_samples - 1) < arr.shape[-1] and not suppress_warning:
        spr = compute_suggested_patch_radius(arr, patch_size)
        warn(
            dimensionality_problem_message(arr, num_samples, spr),
            UserWarning,
            stacklevel=2,
        )

    if isinstance(sigma, np.ndarray):
        var = sigma**2
        if not sigma.shape == arr.shape[:-1]:
            e_s = f"You provided a sigma array with shape {sigma.shape} for data with "
            e_s += f"shape {arr.shape}. Please provide a sigma array "
            e_s += "that matches the spatial dimensions of the data."
            raise ValueError(e_s)
    elif isinstance(sigma, (int, float)):
        var = sigma**2 * np.ones(arr.shape[:-1], dtype=real_dtype)

    dim = arr.shape[-1]
    if tau_factor is None:
        # For complex data, tau_factor needs adjustment
        tau_factor = 1 + np.sqrt(2 * dim / num_samples)

    # Initialize arrays with complex dtype
    theta = np.zeros(arr.shape[:-1], dtype=real_dtype)
    thetax = np.zeros(arr.shape, dtype=calc_dtype)

    if return_sigma is True and sigma is None:
        var = np.zeros(arr.shape[:-1], dtype=real_dtype)
        thetavar = np.zeros(arr.shape[:-1], dtype=real_dtype)

    # Loop through all voxels
    for k in range(patch_radius_arr[2], arr.shape[2] - patch_radius_arr[2]):
        for j in range(patch_radius_arr[1], arr.shape[1] - patch_radius_arr[1]):
            for i in range(patch_radius_arr[0], arr.shape[0] - patch_radius_arr[0]):
                if not mask[i, j, k]:
                    continue
                    
                ix1 = i - patch_radius_arr[0]
                ix2 = i + patch_radius_arr[0] + 1
                jx1 = j - patch_radius_arr[1]
                jx2 = j + patch_radius_arr[1] + 1
                kx1 = k - patch_radius_arr[2]
                kx2 = k + patch_radius_arr[2] + 1

                # Extract patch and reshape
                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(num_samples, dim)
                
                # Mean subtraction
                M = np.mean(X, axis=0)
                X = X - M

                if is_svd:
                    # Complex SVD
                    U, S, Vh = np.linalg.svd(X, full_matrices=False)
                    # For complex data: eigenvalues = S^2 / M
                    d = (S**2) / X.shape[0]
                    # Eigenvectors from V
                    W = Vh.conj().T
                else:
                    # Complex covariance matrix
                    C = (X.conj().T @ X) / X.shape[0]
                    # Eigendecomposition of Hermitian matrix
                    d, W = np.linalg.eigh(C)
                    # Sort in descending order
                    idx = np.argsort(d)[::-1]
                    d = d[idx]
                    W = W[:, idx]

                if sigma is None:
                    # Use complex Wishart classifier
                    this_var, _ = _pca_classifier_complex(d, num_samples)
                else:
                    this_var = var[i, j, k]

                # Threshold for complex data
                tau = tau_factor**2 * this_var
                mask_sig = d > tau
                ncomps = int(np.sum(mask_sig))
                
                # Zero out non-significant components
                W[:, ~mask_sig] = 0

                # Reconstruct
                Xest = X @ W @ W.conj().T + M
                Xest = Xest.reshape(patch_size[0], patch_size[1], patch_size[2], dim)

                # Weight calculation
                this_theta = 1.0 / (1.0 + dim - ncomps)
                
                # Accumulate results
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest * this_theta
                
                if return_sigma is True and sigma is None:
                    var[ix1:ix2, jx1:jx2, kx1:kx2] += this_var * this_theta
                    thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta

    # Final denoised result
    denoised_arr = np.zeros_like(arr)
    mask_theta = theta > 0
    denoised_arr[mask_theta] = thetax[mask_theta] / theta[mask_theta, np.newaxis]
    denoised_arr[mask == 0] = 0
    
    if return_sigma is True:
        if sigma is None:
            var_out = np.zeros_like(var)
            mask_thetavar = thetavar > 0
            var_out[mask_thetavar] = var[mask_thetavar] / thetavar[mask_thetavar]
            var_out[mask == 0] = 0
            return denoised_arr.astype(out_dtype), np.sqrt(var_out)
        else:
            return denoised_arr.astype(out_dtype), sigma
    else:
        return denoised_arr.astype(out_dtype)


@warning_for_keywords()
def mppca_complex(
    arr,
    *,
    mask=None,
    patch_radius=2,
    pca_method="eig",
    return_sigma=False,
    out_dtype=None,
    suppress_warning=False,
):
    """
    Complex Marchenko-Pastur PCA denoising for complex-valued data
    
    Parameters
    ----------
    arr : ndarray
        4D complex array (x, y, z, measurements)
    mask : ndarray, optional
        Boolean mask
    patch_radius : int or list
        Patch radius for local PCA
    pca_method : str
        'eig' or 'svd' (default: 'eig')
    return_sigma : bool
        Whether to return estimated sigma
    out_dtype : dtype
        Output data type
    suppress_warning : bool
        Whether to suppress warnings
    
    Returns
    -------
    denoised : ndarray
        Denoised complex array
    sigma : ndarray, optional
        Estimated noise standard deviation (if return_sigma=True)
    """
    return genpca_complex(
        arr,
        sigma=None,
        mask=mask,
        patch_radius=patch_radius,
        pca_method=pca_method,
        tau_factor=None,
        return_sigma=return_sigma,
        out_dtype=out_dtype,
        suppress_warning=suppress_warning,
    )