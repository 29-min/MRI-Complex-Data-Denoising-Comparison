import numpy as np
import time
from scipy.linalg import eigh
from scipy.linalg.lapack import dgesvd as svd
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter
import warnings

def complex_as_2d_vector_mppca(data, mask=None, patch_radius=None,
                               return_sigma=False, smooth_sigma=2,
                               edge_alpha=2.0, verbose=True):
    """
    Edge‐weighted Complex-as-2D-Vector MP-PCA
    픽셀별 경계(gradient) 기반 가중치를 적용하여 특징부 선명도 향상
    """
    if verbose:
        print("▶ Edge-weighted Complex-as-2D-Vector MP-PCA 시작...")
    start_time = time.time()

    # 데이터 및 마스크 준비
    data = data.astype(np.complex64)
    X, Y, Z, Ne = data.shape
    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)

    # 패치 반경 결정
    if patch_radius is None:
        patch_radius = 2 if Ne<=6 else (3 if Ne<=12 else 4)
    if verbose:
        print(f"   데이터 shape: {data.shape}, 패치 반경: {patch_radius}")

    # gradient map (에지 강조용 weight_map)
    # magnitude of first echo 사용
    mag0 = np.abs(data[...,0])
    gx, gy, gz = np.gradient(mag0)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    grad_norm = grad / (grad.max()+1e-12)
    weight_map = 1.0 + edge_alpha * grad_norm  # [1,1+alpha]

    # 패치 크기
    pr = patch_radius
    patch_size = (2*pr+1,)*3
    num_samples = np.prod(patch_size)

    # 결과 저장 변수
    theta = np.zeros(data.shape, dtype=np.float32)
    thetax = np.zeros(data.shape, dtype=np.complex64)
    if return_sigma:
        var_map = np.zeros((X,Y,Z), dtype=np.float32)
        thetavar = np.zeros((X,Y,Z), dtype=np.float32)
    # … 앞 부분 동일 …
    for k in range(pr, Z-pr):
        for j in range(pr, Y-pr):
            for i in range(pr, X-pr):
                if not mask[i,j,k]: continue

                # 1) 패치 인덱스 추출
                ix1, ix2 = i-pr, i+pr+1
                jx1, jx2 = j-pr, j+pr+1
                kx1, kx2 = k-pr, k+pr+1

                # 2) 패치 및 grad_patch 계산
                patch     = data[ix1:ix2, jx1:jx2, kx1:kx2]            # (px,py,pz,Ne)
                M         = patch.reshape(-1, Ne)                      # (P,Ne)
                grad_patch= grad_norm[ix1:ix2, jx1:jx2, kx1:kx2].mean() # [0,1]

                # 3) 2D 변환 및 중심화 (Wh 없이)
                X2d   = np.concatenate([M.real, M.imag], axis=1)       # (P,2Ne)
                M2d   = X2d.mean(axis=0, keepdims=True)
                Xc    = X2d - M2d

                # 4) SVD → 고유값 d 계산
                U, S, Vt = svd(Xc, full_matrices=0, compute_uv=1)[:3]
                d = (S**2) / num_samples

                # 5) 기존 MP threshold
                sigma = _estimate_sigma_mp(d, num_samples, 2*Ne)
                base_thresh= _compute_mp_threshold_simple(num_samples, 2*Ne, sigma)

                # === 수정 1 : Adaptive Threshold ===
                alpha = 0.5  # 튜닝 파라미터
                local_thresh = base_thresh * (1 - alpha * grad_patch)

                # === 수정 2 : Eigen‐value Boost ===
                base_n = np.sum(d > local_thresh)
                boost  = int(3 * grad_patch)       # 최대 +3개
                n_signal = max(1, min(2*Ne, base_n + boost))

                # 6) denoise (Sd, Xc_dn 등 기존 코드 그대로)
                Sd = S.copy()
                Sd[n_signal:] = 0
                Xc_dn = U @ np.diag(Sd) @ Vt
                X2d_dn = Xc_dn + M2d
                X_dn = (X2d_dn[:, :Ne] + 1j*X2d_dn[:, Ne:]).reshape(*patch_size, Ne)

                # 6) 가중치 = n_signal (DIPY 방식)
                theta_patch = float(n_signal)

                # 7) 중첩 영역에 누적
                theta[ix1:ix2, jx1:jx2, kx1:kx2]      += theta_patch
                thetax[ix1:ix2, jx1:jx2, kx1:kx2]     += X_dn * theta_patch

                if return_sigma:
                    var_map[ix1:ix2, jx1:jx2, kx1:kx2]   += sigma**2 * theta_patch
                    thetavar[ix1:ix2, jx1:jx2, kx1:kx2]  += theta_patch
    # normalize
    theta[theta==0]=1
    denoised = thetax / theta[..., None]
    denoised[~mask]=0
    if return_sigma:
        thetavar[thetavar==0]=1
        sigma_map = np.sqrt(var_map/thetavar)
        if smooth_sigma>0:
            sigma_map = gaussian_filter(sigma_map, smooth_sigma)
        return denoised, sigma_map
    return denoised

def _estimate_sigma_mp(eigenvalues, M, N):
    """
    Marchenko-Pastur 이론에 기반한 노이즈 추정 (DIPY 스타일)
    """
    # 작은 eigenvalue들로부터 노이즈 추정
    n_noise = max(1, N // 4)  # 상위 1/4만 신호로 가정
    noise_eigs = eigenvalues[-n_noise:]
    
    # Median 사용 (robust 추정)
    sigma_est = np.median(noise_eigs)
    
    # MP correction
    gamma = N / M
    if gamma < 1:
        # Under-sampling correction
        correction = (1 - np.sqrt(gamma))**2
        sigma_est = np.sqrt(sigma_est / correction)
    else:
        sigma_est = np.sqrt(sigma_est)
    
    return sigma_est


def _compute_mp_threshold_simple(M, N, sigma):
    """
    간단한 Marchenko-Pastur threshold (DIPY 스타일)
    """
    gamma = N / M
    
    # MP distribution의 upper edge
    lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
    
    # 경험적 보정 factor
    if M < 100:
        correction_factor = 2.5
    elif M < 500:
        correction_factor = 2.3
    else:
        correction_factor = 2.0
    
    # 최종 threshold
    threshold = lambda_plus * correction_factor
    
    return threshold