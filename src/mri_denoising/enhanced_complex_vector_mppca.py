import numpy as np
import time
from scipy.linalg import eigh
from scipy.linalg.lapack import dgesvd as svd
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter
import warnings

def enhanced_complex_2d_vector_mppca(data, mask=None, patch_radius=None, 
                                    return_sigma=False, smooth_sigma=2, 
                                    enhancement_options=None, verbose=True):
    """
    향상된 복소수 2D 벡터 MP-PCA
    
    Parameters:
    -----------
    data : np.ndarray
        복소수 입력 데이터 (X,Y,Z,Ne)
    mask : np.ndarray, optional
        브레인 마스크 (X,Y,Z)
    patch_radius : int, optional
        패치 반경
    return_sigma : bool
        노이즈 맵 반환 여부
    smooth_sigma : float
        노이즈 맵 스무딩 정도
    enhancement_options : dict
        향상 옵션들:
        - 'adaptive_patch': bool (적응적 패치 크기)
        - 'weighted_reconstruction': bool (가중 재구성)
        - 'phase_alignment': bool (위상 정렬)
        - 'iterative_refinement': bool (반복 개선)
        - 'robust_threshold': bool (robust threshold)
    verbose : bool
        상세 정보 출력
    """
    if enhancement_options is None:
        enhancement_options = {
            'adaptive_patch': True,
            'weighted_reconstruction': True,
            'phase_alignment': True,
            'iterative_refinement': False,
            'robust_threshold': True
        }
    
    if verbose:
        print("▶ Enhanced Complex-as-2D-Vector MP-PCA 시작...")
        print(f"   향상 옵션: {enhancement_options}")
    
    start_time = time.time()
    
    # 데이터 준비
    data = data.astype(np.complex64)
    X, Y, Z, Ne = data.shape
    
    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)
    
    # 위상 정렬 (선택적)
    if enhancement_options.get('phase_alignment', True):
        data = _phase_alignment(data, mask)
    
    # 초기 노이즈 추정
    initial_noise_estimate = _estimate_initial_noise(data, mask)
    
    # 패치 크기 설정
    if patch_radius is None:
        patch_radius = _determine_optimal_patch_radius(Ne, initial_noise_estimate)
    
    if verbose:
        print(f"   데이터 shape: {data.shape}")
        print(f"   기본 패치 반경: {patch_radius}")
        print(f"   초기 노이즈 추정: {initial_noise_estimate:.4f}")
    
    # 패치 설정
    patch_radius_arr = np.array([patch_radius, patch_radius, patch_radius])
    if Z == 1:
        patch_radius_arr[2] = 0

    patch_size = 2 * patch_radius_arr + 1
    num_samples = np.prod(patch_size)
    
    # 결과 배열
    theta = np.zeros(data.shape, dtype=np.float32)
    thetax = np.zeros(data.shape, dtype=np.complex64)
    
    if return_sigma:
        var_map = np.zeros((X, Y, Z), dtype=np.float32)
        thetavar = np.zeros((X, Y, Z), dtype=np.float32)
    
    # 통계 수집
    n_components_list = []
    snr_estimates = []
    
    # 메인 디노이징 루프
    if verbose:
        print("   패치 단위 디노이징 수행 중...")
    
    total_patches = 0
    processed_patches = 0
    
    for k in range(patch_radius_arr[2], Z - patch_radius_arr[2]):
        for j in range(patch_radius_arr[1], Y - patch_radius_arr[1]):
            for i in range(patch_radius_arr[0], X - patch_radius_arr[0]):
                
                if not mask[i, j, k]:
                    continue
                
                total_patches += 1
                
                # 적응적 패치 크기 결정
                if enhancement_options.get('adaptive_patch', True):
                    local_patch_radius = _adaptive_patch_radius(
                        data, i, j, k, patch_radius_arr, mask
                    )
                else:
                    local_patch_radius = patch_radius_arr
                
                # 패치 추출
                ix1, ix2 = i - local_patch_radius[0], i + local_patch_radius[0] + 1
                jx1, jx2 = j - local_patch_radius[1], j + local_patch_radius[1] + 1
                kx1, kx2 = k - local_patch_radius[2], k + local_patch_radius[2] + 1
                
                # 경계 체크
                ix1, ix2 = max(0, ix1), min(X, ix2)
                jx1, jx2 = max(0, jx1), min(Y, jx2)
                kx1, kx2 = max(0, kx1), min(Z, kx2)
                
                # 복소수 패치 데이터
                patch_complex = data[ix1:ix2, jx1:jx2, kx1:kx2]
                local_mask = mask[ix1:ix2, jx1:jx2, kx1:kx2]
                
                # 패치 처리
                patch_shape = patch_complex.shape  # (px, py, pz, Ne)
                
                # 패치를 2D 행렬로 변환 (voxels x echoes)
                # 마지막 차원(echo)은 유지하면서 공간 차원만 flatten
                patch_complex_reshaped = patch_complex.reshape(-1, Ne)  # (px*py*pz, Ne)
                local_mask_flat = local_mask.flatten()  # (px*py*pz,)
                
                # 마스크된 voxel만 선택
                valid_voxels = patch_complex_reshaped[local_mask_flat]  # (n_valid, Ne)
                
                if valid_voxels.shape[0] < Ne * 2:  # 최소 voxel 수 체크
                    continue
                
                try:
                    # 향상된 디노이징
                    X_denoised, n_signal, noise_var, local_snr = _enhanced_denoise_patch(
                        valid_voxels, 
                        valid_voxels.shape[0], 
                        Ne,
                        enhancement_options.get('robust_threshold', True)
                    )
                    
                    n_components_list.append(n_signal)
                    snr_estimates.append(local_snr)
                    processed_patches += 1
                    
                    # 패치 재구성
                    X_denoised_full = np.zeros(patch_complex_reshaped.shape, dtype=np.complex64)
                    X_denoised_full[local_mask_flat] = X_denoised
                    X_denoised_full = X_denoised_full.reshape(patch_shape)
                    
                    # 가중치 계산
                    if enhancement_options.get('weighted_reconstruction', True):
                        weight = _compute_adaptive_weight(
                            n_signal, Ne, local_snr, patch_complex_flat.shape[0]
                        )
                    else:
                        weight = 1.0 / (1.0 + Ne - n_signal)
                    
                    theta[ix1:ix2, jx1:jx2, kx1:kx2] += weight * local_mask
                    thetax[ix1:ix2, jx1:jx2, kx1:kx2] += X_denoised_full * weight
                    
                    if return_sigma:
                        var_map[ix1:ix2, jx1:jx2, kx1:kx2] += noise_var * weight * local_mask
                        thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += weight * local_mask
                    
                except Exception as e:
                    if verbose:
                        warnings.warn(f"패치 ({i},{j},{k}) 처리 실패: {str(e)}")
                    continue
    
    # 최종 집계
    theta[theta == 0] = 1
    denoised_data = thetax / theta
    
    # 반복적 개선 (선택적)
    if enhancement_options.get('iterative_refinement', False) and processed_patches > 0:
        if verbose:
            print("   반복적 개선 수행 중...")
        denoised_data = _iterative_refinement(
            data, denoised_data, mask, patch_radius, verbose=False
        )
    
    # 마스크 적용
    denoised_data[~mask] = 0
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n   완료 시간: {elapsed_time:.1f}초")
        print(f"   처리된 패치: {processed_patches}/{total_patches}")
        if n_components_list:
            print(f"   평균 신호 성분: {np.mean(n_components_list):.1f}")
            print(f"   성분 범위: [{np.min(n_components_list)}, {np.max(n_components_list)}]")
            print(f"   평균 로컬 SNR: {np.mean(snr_estimates):.2f}")
    
    if return_sigma:
        thetavar[thetavar == 0] = 1
        sigma_map = np.sqrt(var_map / thetavar)
        sigma_map[~mask] = 0
        if smooth_sigma > 0:
            sigma_map = gaussian_filter(sigma_map, smooth_sigma)
        return denoised_data, sigma_map
    else:
        return denoised_data


def _enhanced_denoise_patch(X_complex, num_samples, Ne, use_robust_threshold=True):
    """
    향상된 패치 디노이징
    """
    # 복소수를 2D 벡터로 변환
    X_real = np.real(X_complex)
    X_imag = np.imag(X_complex)
    X_2d = np.concatenate([X_real, X_imag], axis=1)
    
    # 평균 제거
    M_2d = np.mean(X_2d, axis=0)
    X_centered = X_2d - M_2d
    
    # SVD 수행
    U, S, Vt = svd(X_centered, full_matrices=0, compute_uv=1)[:3]
    
    # Eigenvalues 계산
    d = (S ** 2) / num_samples
    
    # 향상된 노이즈 추정
    if use_robust_threshold:
        sigma, n_signal = _robust_noise_estimation(d, num_samples, 2*Ne)
    else:
        sigma = _estimate_sigma_mp(d, num_samples, 2*Ne)
        threshold = _compute_mp_threshold_simple(num_samples, 2*Ne, sigma)
        n_signal = np.sum(d > threshold)
    
    n_signal = max(1, min(n_signal, 2*Ne))
    
    # 재구성
    S_denoised = S.copy()
    S_denoised[n_signal:] = 0
    
    # 재구성된 2D 데이터
    X_denoised_2d = U @ np.diag(S_denoised) @ Vt + M_2d
    
    # 2D 벡터를 다시 복소수로 변환
    X_denoised_real = X_denoised_2d[:, :Ne]
    X_denoised_imag = X_denoised_2d[:, Ne:]
    X_denoised = X_denoised_real + 1j * X_denoised_imag
    
    # 노이즈 분산 및 SNR 추정
    noise_var = sigma**2
    signal_power = np.mean(d[:n_signal]) if n_signal > 0 else 0
    local_snr = np.sqrt(signal_power) / (sigma + 1e-10)
    
    # 실제 신호 성분 수는 2로 나눔 (real/imag 쌍)
    n_signal_complex = n_signal // 2
    
    return X_denoised, n_signal_complex, noise_var, local_snr


def _phase_alignment(data, mask):
    """
    위상 정렬 - 첫 번째 echo를 기준으로
    """
    aligned_data = data.copy()
    reference_phase = np.angle(data[..., 0])
    
    for echo in range(1, data.shape[-1]):
        phase_diff = np.angle(data[..., echo]) - reference_phase
        # Wrap to [-pi, pi]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        # Apply correction
        aligned_data[..., echo] = data[..., echo] * np.exp(-1j * phase_diff)
    
    return aligned_data


def _estimate_initial_noise(data, mask):
    """
    초기 노이즈 레벨 추정
    """
    # High-frequency component를 이용한 노이즈 추정
    magnitude = np.abs(data)
    noise_estimates = []
    
    for echo in range(data.shape[-1]):
        mag_echo = magnitude[..., echo]
        # 고주파 성분 추출
        smooth = gaussian_filter(mag_echo, sigma=2.0)
        high_freq = mag_echo - smooth
        # Robust 추정
        noise_est = median_abs_deviation(high_freq[mask].flatten()) * 1.4826
        noise_estimates.append(noise_est)
    
    return np.median(noise_estimates)


def _determine_optimal_patch_radius(Ne, noise_level):
    """
    최적 패치 반경 결정
    """
    # 경험적 규칙
    if Ne <= 4:
        base_radius = 2
    elif Ne <= 8:
        base_radius = 3
    else:
        base_radius = 4
    
    # 노이즈 레벨에 따른 조정
    if noise_level > 0.1:  # High noise
        return base_radius + 1
    else:
        return base_radius


def _adaptive_patch_radius(data, i, j, k, base_radius, mask):
    """
    로컬 특성에 따른 적응적 패치 반경
    """
    # 로컬 신호 강도
    local_region = data[
        max(0, i-5):min(data.shape[0], i+6),
        max(0, j-5):min(data.shape[1], j+6),
        max(0, k-5):min(data.shape[2], k+6)
    ]
    
    local_snr = np.mean(np.abs(local_region)) / (np.std(np.abs(local_region)) + 1e-10)
    
    # SNR에 따른 패치 크기 조정
    if local_snr > 10:
        return base_radius - np.array([1, 1, 0])  # 작은 패치
    elif local_snr < 3:
        return base_radius + np.array([1, 1, 0])  # 큰 패치
    else:
        return base_radius


def _compute_adaptive_weight(n_signal, total_components, local_snr, num_samples):
    """
    적응적 가중치 계산
    """
    # 기본 가중치
    base_weight = 1.0 / (1.0 + total_components - n_signal)
    
    # SNR 기반 조정
    snr_factor = np.tanh(local_snr / 5.0)  # 0~1 사이로 정규화
    
    # 샘플 수 기반 조정
    sample_factor = np.sqrt(num_samples / (total_components * 10))
    sample_factor = np.clip(sample_factor, 0.5, 2.0)
    
    return base_weight * snr_factor * sample_factor


def _robust_noise_estimation(eigenvalues, M, N):
    """
    Robust한 노이즈 추정 및 threshold 결정
    """
    # SURE (Stein's Unbiased Risk Estimate) 기반 방법
    n_components = len(eigenvalues)
    
    # 각 threshold 후보에 대한 risk 계산
    risks = []
    for k in range(1, n_components):
        # k개 성분 유지 시 risk
        signal_var = np.sum(eigenvalues[:k])
        noise_var = np.mean(eigenvalues[k:])
        
        # SURE risk
        risk = -signal_var + 2 * k * noise_var
        risks.append(risk)
    
    # 최소 risk를 가지는 성분 수
    if risks:
        optimal_k = np.argmin(risks) + 1
    else:
        optimal_k = 1
    
    # 노이즈 추정
    sigma = np.sqrt(np.median(eigenvalues[optimal_k:]))
    
    return sigma, optimal_k


def _iterative_refinement(original, denoised, mask, patch_radius, max_iter=2, verbose=False):
    """
    반복적 개선 - residual learning
    """
    current = denoised.copy()
    
    for iteration in range(max_iter):
        # Residual 계산
        residual = original - current
        
        # Residual에서 구조적 정보 추출
        residual_denoised = enhanced_complex_2d_vector_mppca(
            residual, 
            mask=mask, 
            patch_radius=patch_radius,
            enhancement_options={'iterative_refinement': False},
            verbose=False
        )
        
        # 적응적 결합
        alpha = 0.3 / (iteration + 1)  # 점진적으로 감소
        current = current + alpha * residual_denoised
    
    return current


def _whitening_transform(X_2d):
    """
    Whitening 변환 (선택적 전처리)
    """
    # 공분산 행렬
    cov = np.cov(X_2d.T)
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Whitening matrix
    D = np.diag(1.0 / np.sqrt(eigvals + 1e-10))
    W = eigvecs @ D @ eigvecs.T
    
    # Whitened data
    X_whitened = X_2d @ W
    
    return X_whitened, W


# 추가 유틸리티 함수
def compare_methods(noisy_data, mask=None, patch_radius=3):
    """
    다양한 방법 비교
    """
    print("방법별 성능 비교:")
    
    # 1. 기본 2D 벡터 방법
    print("\n1. 기본 2D 벡터 방법...")
    basic_result = complex_as_2d_vector_mppca(
        noisy_data, mask=mask, patch_radius=patch_radius, verbose=False
    )
    
    # 2. 향상된 방법 (모든 옵션)
    print("\n2. 향상된 방법 (모든 옵션)...")
    enhanced_result = enhanced_complex_2d_vector_mppca(
        noisy_data, mask=mask, patch_radius=patch_radius,
        enhancement_options={
            'adaptive_patch': True,
            'weighted_reconstruction': True,
            'phase_alignment': True,
            'iterative_refinement': True,
            'robust_threshold': True
        },
        verbose=False
    )
    
    # 3. 향상된 방법 (일부 옵션)
    print("\n3. 향상된 방법 (핵심 옵션만)...")
    partial_enhanced = enhanced_complex_2d_vector_mppca(
        noisy_data, mask=mask, patch_radius=patch_radius,
        enhancement_options={
            'adaptive_patch': True,
            'weighted_reconstruction': True,
            'phase_alignment': False,
            'iterative_refinement': False,
            'robust_threshold': True
        },
        verbose=False
    )
    
    return basic_result, enhanced_result, partial_enhanced