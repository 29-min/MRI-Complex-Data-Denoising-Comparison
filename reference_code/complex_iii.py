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


def _pca_classifier(L, nvoxels):
    if L.size > nvoxels - 1:
        L = L[-(nvoxels - 1) :]
    # 복소수 전용 var 추정 
    n_noise = max(1, int(0.20 * len(L))) # 이걸 조정해서 tau 조정
    noise_vals = np.sort(L)[:n_noise]
    var = np.mean(noise_vals)   # 여기까지는 동일

    c = L.size - 1
    r = L[c] - L[0] - 2 * np.sqrt((c + 1.0) / nvoxels) * var
    while r > 0:
        var = np.mean(L[:c])
        c = c - 1
        r = L[c] - L[0] - 2 * np.sqrt((c + 1.0) / nvoxels) * var
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
def genpca(
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
    if mask is None:
        # If mask is not specified, use the whole volume
        mask = np.ones_like(arr, dtype=bool)[..., 0]

    if out_dtype is None:
        out_dtype = arr.dtype

    # We retain float64 precision, iff the input is in this precision:
    if arr.dtype == np.float64:
        calc_dtype = np.float64
    # Otherwise, we'll calculate things in float32 (saving memory)
    else:
        calc_dtype = np.float32

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
        raise ValueError(
            "Cannot have only 1 sample,\
                          please increase patch_radius."
        )
    # account for mean subtraction by testing #samples - 1
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
            e_s = "You provided a sigma array with a shape"
            e_s += f"{sigma.shape} for data with"
            e_s += f"shape {arr.shape}. Please provide a sigma array"
            e_s += " that matches the spatial dimensions of the data."
            raise ValueError(e_s)
    elif isinstance(sigma, (int, float)):
        var = sigma**2 * np.ones(arr.shape[:-1])

    dim = arr.shape[-1]
    if tau_factor is None:
        tau_factor = 1 + np.sqrt(dim / num_samples)

    theta = np.zeros(arr.shape, dtype=calc_dtype)
    thetax = np.zeros(arr.shape, dtype=arr.dtype)

    if return_sigma is True and sigma is None:
        var = np.zeros(arr.shape[:-1], dtype=calc_dtype)
        thetavar = np.zeros(arr.shape[:-1], dtype=calc_dtype)

    # loop around and find the 3D patch for each direction at each pixel
    for k in range(patch_radius_arr[2], arr.shape[2] - patch_radius_arr[2]):
        for j in range(patch_radius_arr[1], arr.shape[1] - patch_radius_arr[1]):
            for i in range(patch_radius_arr[0], arr.shape[0] - patch_radius_arr[0]):
                # Shorthand for indexing variables:
                if not mask[i, j, k]:
                    continue
                ix1 = i - patch_radius_arr[0]
                ix2 = i + patch_radius_arr[0] + 1
                jx1 = j - patch_radius_arr[1]
                jx2 = j + patch_radius_arr[1] + 1
                kx1 = k - patch_radius_arr[2]
                kx2 = k + patch_radius_arr[2] + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(num_samples, dim)
                # compute the mean
                M = np.mean(X, axis=0)
                # Upcast the dtype for precision in the SVD
                X = X - M



                U, S, Vh = np.linalg.svd(X, full_matrices=False)
                # singular values S는 real, non-negative
                # eigenvalues d = (S^2 / M)
                d = (S**2) / X.shape[0]
                # eigenvectors W (columns) = Vh^H
                W = Vh.conj().T

                if sigma is None:
                    # Random matrix theory
                    this_var, _ = _pca_classifier(d, num_samples)
                else:
                    # Predefined variance
                    this_var = var[i, j, k]
                tau = tau_factor**2 * this_var
                mask_sig = d > tau
                ncomps = int(np.sum(mask_sig))
                W[:, ~mask_sig] = 0

                Xest = X.dot(W).dot(W.conj().T) + M

                Xest = Xest.reshape(patch_size[0], patch_size[1], patch_size[2], dim)
                # This is equation 3 in Manjon 2013:
                this_theta = 1.0 / (1.0 + dim - ncomps)
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest * this_theta
                if return_sigma is True and sigma is None:
                    var[ix1:ix2, jx1:jx2, kx1:kx2] += this_var * this_theta
                    thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta

    denoised_arr = thetax / theta
    # denoised_arr.clip(min=0, out=denoised_arr) 이 부분 제거
    denoised_arr[mask == 0] = 0
    if return_sigma is True:
        if sigma is None:
            var = var / thetavar
            var[mask == 0] = 0
            return denoised_arr.astype(out_dtype), np.sqrt(var)
        else:
            return denoised_arr.astype(out_dtype), sigma
    else:
        return denoised_arr.astype(out_dtype)


@warning_for_keywords()
def localpca(
    arr,
    *,
    sigma=None,
    mask=None,
    patch_radius=2,
    gtab=None,
    patch_radius_sigma=1,
    pca_method="eig",
    tau_factor=2.3,
    return_sigma=False,
    correct_bias=True,
    out_dtype=None,
    suppress_warning=False,
):
    # check gtab is given, if sigma is not given
    if sigma is None and gtab is None:
        raise ValueError("gtab must be provided if sigma is not given")

    # calculate sigma
    if sigma is None:
        sigma = pca_noise_estimate(
            arr,
            gtab,
            correct_bias=correct_bias,
            patch_radius=patch_radius_sigma,
            images_as_samples=True,
        )

    return genpca(
        arr,
        sigma=sigma,
        mask=mask,
        patch_radius=patch_radius,
        pca_method=pca_method,
        tau_factor=tau_factor,
        return_sigma=return_sigma,
        out_dtype=out_dtype,
        suppress_warning=suppress_warning,
    )


@warning_for_keywords()
def mppca(
    arr,
    *,
    mask=None,
    patch_radius=2,
    pca_method="eig",
    return_sigma=False,
    out_dtype=None,
    suppress_warning=False,
):
    return genpca(
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
