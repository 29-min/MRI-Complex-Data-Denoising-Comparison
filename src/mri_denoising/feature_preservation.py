import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import median_abs_deviation
import warnings

def feature_preserving_enhancements(data, mask=None, initial_denoised=None, options=None):
    """
    특징 보존을 위한 추가 향상 기법들
    
    Parameters:
    -----------
    data : np.ndarray
        원본 복소수 데이터 (X,Y,Z,Ne)
    mask : np.ndarray
        브레인 마스크 (X,Y,Z)
    initial_denoised : np.ndarray
        초기 디노이징 결과
    options : dict
        향상 옵션들
    """
    if options is None:
        options = {
            'edge_preservation': True,
            'structure_enhancement': True,
            'adaptive_filtering': True,
            'multi_scale_fusion': True
        }
    
    enhanced = initial_denoised.copy()
    
    # 1. Edge-Aware Filtering
    if options.get('edge_preservation', True):
        enhanced = edge_aware_filtering(data, enhanced, mask)
    
    # 2. Structure Enhancement
    if options.get('structure_enhancement', True):
        enhanced = structure_enhancement(enhanced, mask)
    
    # 3. Adaptive Filtering
    if options.get('adaptive_filtering', True):
        enhanced = adaptive_detail_preservation(data, enhanced, mask)
    
    # 4. Multi-scale Fusion
    if options.get('multi_scale_fusion', True):
        enhanced = multi_scale_fusion(data, enhanced, mask)
    
    return enhanced


def edge_aware_filtering(original, denoised, mask):
    """Edge-aware filtering으로 경계 보존 + 평평한 영역 밝기 개선"""
    X, Y, Z, Ne = original.shape
    filtered = denoised.copy()
    
    for echo in range(Ne):
        # Magnitude 기반 edge detection
        mag_orig = np.abs(original[..., echo])
        mag_den = np.abs(denoised[..., echo])
        
        # Gradient 계산
        grad_x = sobel(mag_orig, axis=0)
        grad_y = sobel(mag_orig, axis=1)
        grad_z = sobel(mag_orig, axis=2) if Z > 1 else np.zeros_like(grad_x)
        
        # Edge strength
        edge_strength = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        edge_strength = gaussian_filter(edge_strength, sigma=1.0)
        
        # Normalize
        edge_strength = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min() + 1e-10)
        
        # *** 가장 효과적인 방법 적용 ***
        # adaptive_alpha 함수 인라인으로 구현
        alpha = np.zeros_like(edge_strength)
        
        # Strong edges (> 0.5): 원본 60-70% 사용
        strong_edge_mask = edge_strength > 0.5
        alpha[strong_edge_mask] = 0.6 + 0.1 * edge_strength[strong_edge_mask]
        
        # Medium edges (0.2-0.5): 적절히 혼합
        medium_edge_mask = (edge_strength > 0.2) & (edge_strength <= 0.5)
        alpha[medium_edge_mask] = 0.3 + 0.6 * edge_strength[medium_edge_mask]
        
        # Flat regions (<= 0.2): 디노이즈 위주 + 밝기 보정
        flat_mask = edge_strength <= 0.2
        alpha[flat_mask] = 0.2 + 0.5 * edge_strength[flat_mask]
        
        # Flat 영역 밝기 보정
        brightness_boost = np.ones_like(mag_den)
        brightness_boost[flat_mask] = 1.3  # 평평한 영역 30% 밝게
        brightness_boost = gaussian_filter(brightness_boost, sigma=2.0)  # 부드러운 전환
        
        # Complex 도메인에서 blending
        filtered_complex = (1 - alpha) * denoised[..., echo] + alpha * original[..., echo]
        
        # Magnitude에 brightness boost 적용
        filtered_mag = np.abs(filtered_complex) * brightness_boost
        filtered_phase = np.angle(filtered_complex)
        
        filtered[..., echo] = filtered_mag * np.exp(1j * filtered_phase)
    
    # 마스크 적용
    filtered[~mask] = 0
    
    return filtered


def structure_enhancement(denoised, mask):
    """
    구조적 특징 강화
    """
    X, Y, Z, Ne = denoised.shape
    enhanced = denoised.copy()
    
    for echo in range(Ne):
        # Magnitude
        mag = np.abs(denoised[..., echo])
        phase = np.angle(denoised[..., echo])
        
        # 1. Unsharp masking for magnitude
        smooth = gaussian_filter(mag, sigma=2.0)
        detail = mag - smooth
        
        # Adaptive enhancement factor - 간단한 방법
        # 전체 이미지의 contrast를 기준으로 함
        global_std = np.std(mag[mask]) if mask.sum() > 0 else np.std(mag)
        
        # 로컬 contrast 계산 (sliding window)
        from scipy.ndimage import generic_filter
        
        def local_std(values):
            return np.std(values)
        
        # 로컬 표준편차 계산 (7x7x7 window)
        local_contrast = generic_filter(mag, local_std, size=(7, 7, 7 if Z > 1 else 1))
        
        # Enhancement factor 계산
        enhancement_factor = 1.0 + 0.5 * np.tanh(local_contrast / (global_std + 1e-10))
        
        # Enhanced magnitude
        mag_enhanced = mag + enhancement_factor * detail * 0.3
        
        # 2. Phase coherence enhancement
        phase_smooth = gaussian_filter(np.cos(phase), sigma=1.0) + 1j * gaussian_filter(np.sin(phase), sigma=1.0)
        phase_smooth = np.angle(phase_smooth)
        
        # Reconstruct
        enhanced[..., echo] = mag_enhanced * np.exp(1j * phase_smooth)
    
    enhanced[~mask] = 0
    return enhanced


def adaptive_detail_preservation(original, denoised, mask):
    """
    적응적 디테일 보존
    """
    X, Y, Z, Ne = original.shape
    result = denoised.copy()
    
    # Local SNR 추정
    for echo in range(Ne):
        mag_orig = np.abs(original[..., echo])
        mag_den = np.abs(denoised[..., echo])
        
        # 로컬 영역별 SNR 추정 (패치 기반)
        patch_size = 5
        for i in range(0, X, patch_size):
            for j in range(0, Y, patch_size):
                for k in range(0, Z, patch_size):
                    # 패치 경계
                    i_end = min(i + patch_size, X)
                    j_end = min(j + patch_size, Y)
                    k_end = min(k + patch_size, Z)
                    
                    # 로컬 패치
                    local_orig = mag_orig[i:i_end, j:j_end, k:k_end]
                    local_den = mag_den[i:i_end, j:j_end, k:k_end]
                    local_mask = mask[i:i_end, j:j_end, k:k_end]
                    
                    if local_mask.sum() == 0:
                        continue
                    
                    # 로컬 SNR
                    signal = np.mean(local_den[local_mask])
                    noise = np.std(local_orig[local_mask] - local_den[local_mask])
                    local_snr = signal / (noise + 1e-10)
                    
                    # SNR 기반 가중치
                    if local_snr > 10:  # High SNR: 디테일 보존
                        weight = 0.3
                    elif local_snr < 3:  # Low SNR: 강한 디노이징
                        weight = 0.0
                    else:
                        weight = 0.3 * (local_snr - 3) / 7
                    
                    # 적응적 블렌딩
                    result[i:i_end, j:j_end, k:k_end, echo] = (
                        (1 - weight) * denoised[i:i_end, j:j_end, k:k_end, echo] +
                        weight * original[i:i_end, j:j_end, k:k_end, echo]
                    )
    
    result[~mask] = 0
    return result


def multi_scale_fusion(original, denoised, mask):
    """
    다중 스케일 융합
    """
    X, Y, Z, Ne = original.shape
    
    # 다양한 스케일에서 특징 추출
    scales = [1, 2, 4]
    features = []
    
    for echo in range(Ne):
        echo_features = []
        mag_orig = np.abs(original[..., echo])
        mag_den = np.abs(denoised[..., echo])
        
        for scale in scales:
            # 각 스케일에서 특징 추출
            if scale == 1:
                feature = mag_den
            else:
                # Downsampling → processing → upsampling
                down = gaussian_filter(mag_den, sigma=scale/2)
                feature = mag_den - down  # Detail at this scale
            
            echo_features.append(feature)
        
        # 스케일별 가중치 (fine to coarse)
        weights = [0.5, 0.3, 0.2]
        
        # 융합
        fused = np.zeros_like(mag_den)
        for feat, w in zip(echo_features, weights):
            fused += w * feat
        
        # Phase 유지하면서 magnitude 업데이트
        phase = np.angle(denoised[..., echo])
        denoised[..., echo] = fused * np.exp(1j * phase)
    
    denoised[~mask] = 0
    return denoised


def bilateral_complex_filter(data, spatial_sigma=1.0, range_sigma=0.1):
    """
    복소수 데이터를 위한 bilateral filter
    """
    X, Y, Z, Ne = data.shape
    filtered = np.zeros_like(data)
    
    # 각 echo별로 처리
    for echo in range(Ne):
        # Magnitude와 phase 분리
        mag = np.abs(data[..., echo])
        phase = np.angle(data[..., echo])
        
        # Magnitude에 bilateral filter 적용
        mag_filtered = bilateral_filter_3d(mag, spatial_sigma, range_sigma)
        
        # Phase smoothing (optional)
        phase_cos = gaussian_filter(np.cos(phase), sigma=spatial_sigma*0.5)
        phase_sin = gaussian_filter(np.sin(phase), sigma=spatial_sigma*0.5)
        phase_filtered = np.arctan2(phase_sin, phase_cos)
        
        # 재결합
        filtered[..., echo] = mag_filtered * np.exp(1j * phase_filtered)
    
    return filtered


def bilateral_filter_3d(image, spatial_sigma, range_sigma):
    """
    3D bilateral filter 구현 (간단한 버전)
    """
    from scipy.ndimage import gaussian_filter
    
    # 간단한 근사: edge-preserving smoothing
    # 실제 구현시에는 더 정교한 bilateral filter 사용
    smooth = gaussian_filter(image, spatial_sigma)
    diff = image - smooth
    
    # Range kernel (intensity similarity)
    weight = np.exp(-diff**2 / (2 * range_sigma**2))
    
    # Weighted average
    filtered = smooth + weight * diff
    
    return filtered


def optimize_snr_with_features(original, initial_denoised, mask, verbose=True):
    """
    SNR 최적화를 위한 통합 처리
    """
    if verbose:
        print("▶ 특징 보존 강화 처리 시작...")
    
    # 1단계: Edge-aware filtering
    if verbose:
        print("  1. Edge-aware filtering...")
    result = edge_aware_filtering(original, initial_denoised, mask)
    
    # 2단계: Structure enhancement
    if verbose:
        print("  2. Structure enhancement...")
    result = structure_enhancement(result, mask)
    
    # 3단계: Adaptive detail preservation
    if verbose:
        print("  3. Adaptive detail preservation...")
    result = adaptive_detail_preservation(original, result, mask)
    
    # 4단계: Multi-scale fusion (선택적)
    if verbose:
        print("  4. Multi-scale fusion...")
    result = multi_scale_fusion(original, result, mask)
    
    if verbose:
        print("▶ 특징 보존 강화 완료!")
    
    return result


# 메인 함수에 통합하기 위한 wrapper
def enhanced_mppca_with_features(data, mask=None, patch_radius=3, 
                                verbose=True, **kwargs):
    """
    특징 보존이 강화된 MP-PCA
    """
    # 기본 enhanced MP-PCA 실행
    from .enhanced_complex_vector_mppca import enhanced_complex_2d_vector_mppca
    
    initial_denoised, sigma_map = enhanced_complex_2d_vector_mppca(
        data, mask=mask, patch_radius=patch_radius, 
        return_sigma=True, verbose=verbose, **kwargs
    )
    
    # 특징 보존 강화 적용
    final_denoised = optimize_snr_with_features(
        data, initial_denoised, mask, verbose=verbose
    )
    
    return final_denoised, sigma_map
