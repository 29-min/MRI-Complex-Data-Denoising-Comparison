import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO)

def _robust_sigma_estimation(eigvals, gamma, n_noise_ratio=0.1):
    """
    로버스트한 잡음 표준편차 추정: 최하위 eigenvalue들의 MAD 사용
    """
    N = len(eigvals)
    n_noise = max(1, int(np.ceil(n_noise_ratio * N)))
    noise_vals = np.sort(eigvals)[:n_noise]
    sigma_init = np.sqrt(median_abs_deviation(noise_vals)**2)
    lambda_plus = (1 + np.sqrt(gamma))**2
    bulk_max = sigma_init**2 * lambda_plus * 1.05
    bulk_vals = eigvals[eigvals <= bulk_max]
    if len(bulk_vals) >= n_noise:
        sigma = np.sqrt(np.mean(np.sort(bulk_vals)[:n_noise]))
    else:
        sigma = sigma_init
    if gamma < 1:
        sigma *= np.sqrt(1/(1-gamma))
    return sigma


def _denoise_patch_svd(X_patch, M, N, threshold_method='adaptive'):
    """
    패치별 복소수 SVD 기반 MP-PCA 디노이징
    threshold_method:
      - 'mp': MP 이론 임계값 (τ = σ²·λ₊)
      - 'adaptive': 로컬 SNR 기반 τ 동적 조절
      - 'gap': singular value 간 최대 갭 기준
    """
    # 중심화 및 채널 분산 정규화
    X_mean = X_patch.mean(axis=0, keepdims=True)
    chan_sigma = np.std(X_patch, axis=0, keepdims=True) + 1e-12
    Xn = (X_patch - X_mean) / chan_sigma

    # 복소수 SVD
    U, S, Vh = np.linalg.svd(Xn, full_matrices=False)

    # 잡음 σ 추정
    gamma = N / M
    sigma = _robust_sigma_estimation(S**2, gamma)
    lambda_plus = (1 + np.sqrt(gamma))**2

    # 임계값 결정
    if threshold_method == 'mp':
        tau = sigma**2 * lambda_plus
        mask_sig = S > np.sqrt(tau)
    elif threshold_method == 'adaptive':
        signal_power = np.mean(np.abs(X_patch)**2)
        noise_power = np.var(np.abs(X_patch) - np.abs(X_patch).mean())
        local_snr = signal_power / (noise_power + 1e-12)
        
        target_snr = 9.0
        snr_norm = np.clip(local_snr / target_snr, 0, 1)
        tau_factor = 1.0 + 0.7 * (1 - snr_norm)
        tau = tau_factor * sigma**2 * lambda_plus
        mask_sig = S > np.sqrt(tau)
    else:
        diffs = S[:-1] - S[1:]
        k = np.argmax(diffs) + 1
        mask_sig = np.zeros_like(S, dtype=bool)
        mask_sig[:k] = True

    # 신호 singular component만 유지
    S_thr = np.where(mask_sig, S, 0)
    Xd = (U * S_thr) @ Vh
    Xd = Xd * chan_sigma + X_mean
    return Xd, sigma


def improved_complex_wishart_mppca(
    data, mask, patch_radius=2, n_jobs=4,
    return_sigma=False, threshold_method='gap'
):
    """
    SVD 기반 복소 MP-PCA 디노이징
    threshold_method: 'mp', 'adaptive', 'gap'
    """
    X, Y, Z, N = data.shape
    pr = patch_radius
    size = 2*pr + 1
    M = size**3

    # sliding window view
    patches = sliding_window_view(data, window_shape=(size, size, size, N))

    denoised = np.zeros_like(data)
    weight = np.zeros((X, Y, Z), dtype=np.float32)
    sigma_map = np.zeros((X, Y, Z), dtype=np.float32) if return_sigma else None

    coords = [
        (i, j, k)
        for i in range(pr, X-pr)
        for j in range(pr, Y-pr)
        for k in range(pr, Z-pr)
        if mask[i, j, k]
    ]

    def process_voxel(i, j, k):
        patch = patches[i-pr, j-pr, k-pr].reshape(M, N)
        den_patch, sigma = _denoise_patch_svd(patch, M, N, threshold_method)
        return i, j, k, den_patch.reshape(size, size, size, N), sigma

    # threading backend 사용하여 함수 직렬화 문제 회피
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_voxel)(i, j, k) for i, j, k in coords
    )

    for i, j, k, patch_denoised, sigma in results:
        denoised[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += patch_denoised
        weight[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += 1
        if return_sigma:
            sigma_map[i, j, k] = sigma

    weight[weight == 0] = 1
    denoised = denoised / weight[..., None]

    denoised[~mask] = data[~mask]
    if return_sigma:
        sigma_map = gaussian_filter(sigma_map, sigma=1.0)
        return denoised, sigma_map
    return denoised
