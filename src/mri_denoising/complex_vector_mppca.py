import numpy as np
import time
from scipy.linalg import eigh
from scipy.linalg.lapack import dgesvd as svd
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter
import warnings

def complex_as_2d_vector_mppca(data, mask=None, patch_radius=None, return_sigma=False, 
                               smooth_sigma=2, verbose=True):
    """
    복소수 데이터를 2D 벡터로 처리하는 MP-PCA
    
    복소수를 [real, imag] 2차원 벡터로 변환하여 처리
    
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
    verbose : bool
        상세 정보 출력
        
    Returns:
    --------
    denoised_data : np.ndarray
        디노이징된 복소수 데이터
    sigma_map : np.ndarray, optional
        노이즈 표준편차 맵
    """
    if verbose:
        print("▶ Complex-as-2D-Vector MP-PCA 시작...")
    start_time = time.time()
    
    # 데이터 준비
    data = data.astype(np.complex64)
    X, Y, Z, Ne = data.shape
    
    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)
    
    # 패치 크기 설정
    if patch_radius is None:
        if Ne <= 6:
            patch_radius = 2  # 5x5x5 = 125 voxels
        elif Ne <= 12:
            patch_radius = 3  # 7x7x7 = 343 voxels
        else:
            patch_radius = 4  # 9x9x9 = 729 voxels
    
    if verbose:
        print(f"   데이터 shape: {data.shape}")
        print(f"   패치 반경: {patch_radius}")
    
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
                
                # 패치 추출
                ix1, ix2 = i - patch_radius_arr[0], i + patch_radius_arr[0] + 1
                jx1, jx2 = j - patch_radius_arr[1], j + patch_radius_arr[1] + 1
                kx1, kx2 = k - patch_radius_arr[2], k + patch_radius_arr[2] + 1
                
                # 복소수 패치 데이터
                patch_complex = data[ix1:ix2, jx1:jx2, kx1:kx2]  # (px, py, pz, Ne)
                
                # 패치를 2D 행렬로 변환 - echo 차원은 유지
                patch_reshaped = patch_complex.reshape(-1, Ne)  # (px*py*pz, Ne)
                
                try:
                    # 복소수를 2D 벡터로 변환하여 디노이징
                    X_denoised, n_signal, noise_var = _denoise_patch_2d_vector(
                        patch_reshaped, patch_reshaped.shape[0], Ne
                    )
                    
                    n_components_list.append(n_signal)
                    processed_patches += 1
                    
                    # 패치 재구성
                    X_denoised = X_denoised.reshape(patch_size[0], patch_size[1], patch_size[2], Ne)
                    
                    # 가중치 계산
                    this_theta = 1.0 / (1.0 + Ne - n_signal)
                    
                    theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                    thetax[ix1:ix2, jx1:jx2, kx1:kx2] += X_denoised * this_theta
                    
                    if return_sigma:
                        var_map[ix1:ix2, jx1:jx2, kx1:kx2] += noise_var * this_theta
                        thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                    
                except Exception as e:
                    if verbose:
                        warnings.warn(f"패치 ({i},{j},{k}) 처리 실패: {str(e)}")
                    continue
    
    # 최종 집계
    theta[theta == 0] = 1
    denoised_data = thetax / theta
    
    # 마스크 적용
    denoised_data[~mask] = 0
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n   완료 시간: {elapsed_time:.1f}초")
        print(f"   처리된 패치: {processed_patches}/{total_patches}")
        if n_components_list:
            print(f"   평균 신호 성분: {np.mean(n_components_list):.1f}")
            print(f"   성분 범위: [{np.min(n_components_list)}, {np.max(n_components_list)}]")
    
    if return_sigma:
        thetavar[thetavar == 0] = 1
        sigma_map = np.sqrt(var_map / thetavar)
        sigma_map[~mask] = 0
        if smooth_sigma > 0:
            sigma_map = gaussian_filter(sigma_map, smooth_sigma)
        return denoised_data, sigma_map
    else:
        return denoised_data


def _denoise_patch_2d_vector(X_complex, num_samples, Ne):
    """
    복소수를 2D 벡터로 변환하여 패치 디노이징
    
    복소수 데이터를 [real, imag] 형태의 2차원 벡터로 변환하여
    PCA를 수행합니다.
    """
    # 복소수를 2D 벡터로 변환
    # X_complex shape: (num_samples, Ne)
    # X_2d shape: (num_samples, 2*Ne)
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
    
    # 노이즈 추정 (DIPY 스타일)
    sigma = _estimate_sigma_mp(d, num_samples, 2*Ne)
    
    # Marchenko-Pastur threshold 계산
    threshold = _compute_mp_threshold_simple(num_samples, 2*Ne, sigma)
    
    # 신호 성분 결정
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
    
    # 노이즈 분산
    noise_var = sigma**2
    
    # 실제 신호 성분 수는 2로 나눔 (real/imag 쌍)
    n_signal_complex = n_signal // 2
    
    return X_denoised, n_signal_complex, noise_var


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


# 추가 유틸리티 함수들
def complex_to_2d_concatenated_mppca(data, mask=None, patch_radius=None, 
                                    return_sigma=False, verbose=True):
    """
    다른 방식: 모든 echo의 real/imag를 연결하여 처리
    
    예: 6 echo 복소수 데이터 -> 12차원 실수 데이터로 변환
    """
    if verbose:
        print("▶ Complex-to-Concatenated MP-PCA 시작...")
    
    # 복소수 데이터를 실수로 변환
    data_real = np.real(data)
    data_imag = np.imag(data)
    
    # Echo 차원에서 연결
    data_concatenated = np.concatenate([data_real, data_imag], axis=-1)
    
    # 표준 MP-PCA 적용 (실수 데이터로)
    from dipy.denoise.localpca import mppca
    
    if return_sigma:
        denoised_concat, sigma = mppca(data_concatenated, mask=mask, 
                                      patch_radius=patch_radius, 
                                      return_sigma=True)
        
        # 다시 복소수로 변환
        Ne = data.shape[-1]
        denoised_real = denoised_concat[..., :Ne]
        denoised_imag = denoised_concat[..., Ne:]
        denoised_complex = denoised_real + 1j * denoised_imag
        
        return denoised_complex, sigma
    else:
        denoised_concat = mppca(data_concatenated, mask=mask, 
                               patch_radius=patch_radius)
        
        # 다시 복소수로 변환
        Ne = data.shape[-1]
        denoised_real = denoised_concat[..., :Ne]
        denoised_imag = denoised_concat[..., Ne:]
        denoised_complex = denoised_real + 1j * denoised_imag
        
        return denoised_complex


def test_2d_vector_approach():
    """
    2D 벡터 접근법 테스트
    """
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 복소수 신호 생성
    X, Y, Z, Ne = 50, 50, 20, 6
    
    # 진짜 신호 (낮은 rank)
    true_signal = np.zeros((X, Y, Z, Ne), dtype=np.complex64)
    for i in range(3):  # 3개의 주요 성분
        spatial = np.random.randn(X, Y, Z)
        temporal = np.random.randn(Ne) + 1j * np.random.randn(Ne)
        temporal /= np.abs(temporal)  # 정규화
        
        for e in range(Ne):
            true_signal[..., e] += spatial * temporal[e] * (10 - i*3)
    
    # 노이즈 추가
    noise_level = 2.0
    noise = noise_level * (np.random.randn(X, Y, Z, Ne) + 
                          1j * np.random.randn(X, Y, Z, Ne))
    
    noisy_data = true_signal + noise
    
    # 마스크 생성
    center = [X//2, Y//2, Z//2]
    mask = np.zeros((X, Y, Z), dtype=bool)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 20:
                    mask[i, j, k] = True
    
    # 2D 벡터 방식으로 디노이징
    print("2D 벡터 방식 테스트...")
    denoised, sigma = complex_as_2d_vector_mppca(noisy_data, mask=mask, 
                                                 patch_radius=2, 
                                                 return_sigma=True,
                                                 verbose=True)
    
    # 결과 평가
    mse_noisy = np.mean(np.abs(noisy_data[mask] - true_signal[mask])**2)
    mse_denoised = np.mean(np.abs(denoised[mask] - true_signal[mask])**2)
    
    print(f"\n결과:")
    print(f"  노이즈 MSE: {mse_noisy:.4f}")
    print(f"  디노이즈 MSE: {mse_denoised:.4f}")
    print(f"  개선율: {(1 - mse_denoised/mse_noisy)*100:.1f}%")
    print(f"  추정된 노이즈 레벨: {np.mean(sigma[mask]):.4f} (실제: {noise_level:.4f})")
    
    return denoised, sigma


if __name__ == "__main__":
    # 테스트 실행
    denoised, sigma = test_2d_vector_approach()