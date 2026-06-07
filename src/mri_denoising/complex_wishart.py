import numpy as np
import time
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
import warnings

def complex_wishart_mppca(data, mask=None, patch_radius=None, 
                         tau_factor=None, return_sigma=False, 
                         phase_correct=True, verbose=True):
    """
    Complex-valued MP-PCA denoising using Complex Wishart distribution theory.
    Based on DIPY's implementation but adapted for complex-valued data.
    
    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N_echoes)
        Complex-valued 4D data array
    mask : ndarray, shape (X, Y, Z), optional
        Boolean mask for brain region
    patch_radius : int, optional
        Radius of sliding window. Default: auto-select based on N_echoes
    tau_factor : float, optional
        Factor controlling threshold (tau = tau_factor * sigma * sqrt(lambda_plus))
        If None, automatically computed using MP theory. Default: None
    return_sigma : bool, optional
        If True, return noise standard deviation map
    phase_correct : bool, optional
        Apply phase correction preprocessing
    verbose : bool, optional
        Print progress information
        
    Returns
    -------
    denoised : ndarray
        Denoised complex data
    sigma : ndarray (optional)
        Noise standard deviation map (if return_sigma=True)
        
    References
    ----------
    Veraart et al. "Denoising of diffusion MRI using random matrix theory"
    NeuroImage 142 (2016): 394-406
    """
    if verbose:
        print("▶ Complex Wishart MP-PCA Denoising")
        start_time = time.time()
    
    # Ensure complex data
    if not np.iscomplexobj(data):
        raise ValueError("Input data must be complex-valued")
    
    # Get dimensions
    X, Y, Z, N = data.shape
    
    # Create mask if not provided
    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)
    
    # Auto-select patch radius based on DIPY logic
    if patch_radius is None:
        # Ensure M > N for MP theory validity
        if N <= 10:
            patch_radius = 3  # 7x7x7 = 343 voxels
        elif N <= 20:
            patch_radius = 2  # 5x5x5 = 125 voxels
        else:
            patch_radius = 1  # 3x3x3 = 27 voxels
            
    if verbose:
        print(f"   Data shape: {data.shape}")
        print(f"   Patch radius: {patch_radius} (patch size: {2*patch_radius+1}³)")
    
    # Phase correction preprocessing (optional but recommended)
    if phase_correct:
        data = _apply_phase_correction(data, mask, verbose)
    
    # Initialize output arrays
    denoised = np.zeros_like(data)
    weights = np.zeros((X, Y, Z), dtype=np.float32)
    
    if return_sigma:
        sigma_map = np.zeros((X, Y, Z), dtype=np.float32)
        sigma_weights = np.zeros((X, Y, Z), dtype=np.float32)
    
    # Patch parameters
    pr = patch_radius
    patch_size = 2*pr + 1
    M = patch_size**3  # number of voxels in patch
    
    # Check MP theory validity
    if M <= N:
        warnings.warn(f"Patch size ({M} voxels) should be larger than measurements ({N}). "
                     f"Consider increasing patch_radius.")
    
    # Progress tracking
    total_voxels = np.sum(mask)
    processed = 0
    
    # Main denoising loop
    for k in range(pr, Z-pr):
        for j in range(pr, Y-pr):
            for i in range(pr, X-pr):
                if not mask[i, j, k]:
                    continue
                
                # Extract patch
                patch = data[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1]
                
                # Reshape to matrix form (M voxels x N measurements)
                X_patch = patch.reshape(M, N)
                
                # Denoise patch using Complex Wishart MP-PCA
                X_denoised, sigma_est = _complex_wishart_denoise(
                    X_patch, M, N, tau_factor)
                
                # Reshape back to patch
                patch_denoised = X_denoised.reshape(patch_size, patch_size, 
                                                   patch_size, N)
                
                # Accumulate results (overlapping patches)
                denoised[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += patch_denoised
                weights[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += 1.0
                
                if return_sigma:
                    sigma_map[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += sigma_est
                    sigma_weights[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += 1.0
                
                processed += 1
                if verbose and processed % 1000 == 0:
                    percent = 100 * processed / total_voxels
                    print(f"   Progress: {percent:.1f}% ({processed}/{total_voxels} voxels)")
    
    # Normalize by overlap weights
    weights[weights == 0] = 1.0
    denoised = denoised / weights[..., None]
    
    # Apply mask
    denoised[~mask] = 0
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"   ✓ Completed in {elapsed:.1f} seconds")
    
    if return_sigma:
        sigma_weights[sigma_weights == 0] = 1.0
        sigma_map = sigma_map / sigma_weights
        # Apply Gaussian smoothing to sigma map (as in DIPY)
        sigma_map = gaussian_filter(sigma_map, sigma=1.0)
        return denoised, sigma_map
    
    return denoised


def _apply_phase_correction(data, mask, verbose=False):
    """
    Apply phase correction to remove low-frequency phase variations.
    This improves the effectiveness of MP-PCA on complex data.
    """
    if verbose:
        print("   Applying phase correction...")
    
    # Extract magnitude and phase
    magnitude = np.abs(data)
    phase = np.angle(data)
    
    # Estimate and remove smooth phase variations
    phase_smooth = np.zeros_like(phase)
    for echo in range(data.shape[-1]):
        # Smooth phase using Gaussian filter
        phase_echo = phase[..., echo].copy()
        phase_echo[~mask] = 0  # Mask out non-brain regions
        phase_smooth[..., echo] = gaussian_filter(phase_echo, sigma=5.0)
    
    # Remove smooth phase component
    phase_corrected = phase - phase_smooth
    
    # Reconstruct complex data with corrected phase
    return magnitude * np.exp(1j * phase_corrected)


def _complex_wishart_denoise(X, M, N, tau_factor=None):
    """
    Denoise a patch using Complex Wishart MP-PCA theory.
    
    Parameters
    ----------
    X : ndarray, shape (M, N)
        Complex-valued data matrix (M voxels x N measurements)
    M : int
        Number of voxels (samples)
    N : int
        Number of measurements (features)
    tau_factor : float, optional
        Threshold factor. If None, compute automatically
        
    Returns
    -------
    X_denoised : ndarray, shape (M, N)
        Denoised patch
    sigma : float
        Estimated noise standard deviation
    """
    # Mean center the data
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean
    
    # Compute complex sample covariance matrix
    # C = (1/M) * X^H * X (Hermitian matrix)
    C = (X_centered.conj().T @ X_centered) / M
    
    # Eigendecomposition of Hermitian matrix
    # eigenvalues are real, eigenvectors are complex
    eigenvalues, eigenvectors = eigh(C)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Estimate noise standard deviation using MP theory
    sigma = _estimate_sigma_complex_mp(eigenvalues, M, N)
    
    # Compute threshold using Complex MP distribution
    if tau_factor is None:
        # Automatic threshold selection based on MP theory
        tau_factor = _compute_tau_factor_complex(M, N)
    
    # MP distribution parameters
    gamma = N / M
    lambda_plus = (1 + np.sqrt(gamma))**2
    
    # Threshold for eigenvalue selection
    tau = tau_factor * sigma**2 * lambda_plus
    
    # Select signal components
    n_components = np.sum(eigenvalues > tau)
    
    # Ensure at least one component
    n_components = max(1, min(N, n_components))
    
    # Reconstruct using only signal components
    if n_components < N:
        # Project onto signal subspace
        U_signal = eigenvectors[:, :n_components]
        # X_proj = X_centered * U_signal * U_signal^H
        X_proj = X_centered @ U_signal @ U_signal.conj().T
        X_denoised = X_proj + X_mean
    else:
        # Keep all components (no denoising needed)
        X_denoised = X
    
    return X_denoised, sigma


def _estimate_sigma_complex_mp(eigenvalues, M, N):
    """
    Estimate noise standard deviation using Complex Wishart MP theory.
    
    Uses the bulk of the eigenvalue distribution for robust estimation,
    following DIPY's approach but adapted for complex data.
    """
    # Parameters for MP distribution
    gamma = N / M
    
    # Expected range of noise eigenvalues under MP law
    lambda_minus = (1 - np.sqrt(gamma))**2
    lambda_plus = (1 + np.sqrt(gamma))**2
    
    # Initial sigma estimate using median of smaller eigenvalues
    # Use bottom 10% of eigenvalues for initial estimate
    n_noise = max(1, int(0.1 * N))
    noise_eigenvalues = eigenvalues[-n_noise:]
    
    # Median-based estimator (robust to outliers)
    sigma_init = np.sqrt(np.median(noise_eigenvalues))
    
    # Iterative refinement (optional, following DIPY approach)
    # Find eigenvalues within MP bulk
    bulk_max = sigma_init**2 * lambda_plus * 1.1  # 10% margin
    bulk_eigenvalues = eigenvalues[eigenvalues <= bulk_max]
    
    if len(bulk_eigenvalues) > 0:
        # Refined estimate using eigenvalues in bulk
        sigma = np.sqrt(np.mean(bulk_eigenvalues[-n_noise:]))
    else:
        sigma = sigma_init
    
    # Apply finite sample correction for complex case
    if gamma < 1:
        # Complex Wishart has different finite sample behavior
        correction = 1.0 / (1 - gamma)
        sigma = sigma * np.sqrt(correction)
    
    return sigma


def _compute_tau_factor_complex(M, N):
    """
    Compute tau_factor for Complex Wishart case based on matrix dimensions.
    
    Following DIPY's approach but adjusted for complex data which has
    better statistical properties and needs less aggressive thresholding.
    """
    gamma = N / M
    
    # Base tau_factor (empirically determined)
    # Complex case needs lower values than real case
    if M < 50:
        base_tau = 2.0
    elif M < 100:
        base_tau = 1.8
    elif M < 500:
        base_tau = 1.5
    else:
        base_tau = 1.2
    
    # Adjust based on gamma (aspect ratio)
    if gamma > 0.5:
        # High gamma: fewer samples relative to features
        tau_factor = base_tau * (1 + 0.5 * gamma)
    else:
        # Low gamma: plenty of samples
        tau_factor = base_tau
    
    return tau_factor


# Utility functions for evaluation
def compute_snr_improvement(original, noisy, denoised, mask=None):
    """Compute SNR improvement from denoising"""
    if mask is None:
        mask = np.ones(original.shape[:3], dtype=bool)
    
    # Signal power (from original)
    signal_power = np.mean(np.abs(original[mask])**2)
    
    # Noise power (original noise)
    noise = noisy - original
    noise_power = np.mean(np.abs(noise[mask])**2)
    
    # Residual noise after denoising
    residual = denoised - original
    residual_power = np.mean(np.abs(residual[mask])**2)
    
    # SNR calculations
    snr_noisy = 10 * np.log10(signal_power / noise_power)
    snr_denoised = 10 * np.log10(signal_power / residual_power)
    
    return {
        'snr_noisy': snr_noisy,
        'snr_denoised': snr_denoised,
        'snr_gain': snr_denoised - snr_noisy
    }


# Example usage
if __name__ == "__main__":
    # Example with your data structure
    print("Complex Wishart MP-PCA Denoising Example\n")
    
    # Simulate multi-echo GRE data
    np.random.seed(42)
    shape = (64, 64, 32, 6)  # 6 echoes
    
    # Create realistic complex signal with T2* decay
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = [s//2 for s in shape[:3]]
    
    # Brain mask
    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
    mask = dist < 20
    
    # Generate complex signal with echo decay
    TE = np.array([5, 10, 15, 20, 25, 30]) * 1e-3  # Echo times in seconds
    T2_star = 30e-3  # T2* = 30ms
    
    signal = np.zeros(shape, dtype=np.complex128)
    for i in range(shape[3]):
        magnitude = mask.astype(float) * np.exp(-TE[i] / T2_star)
        phase = np.random.randn(*shape[:3]) * 0.1  # Small phase variations
        signal[..., i] = magnitude * np.exp(1j * phase)
    
    # Add complex Gaussian noise
    noise_level = 0.05
    noise = noise_level * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    noisy_data = signal + noise
    
    # Apply denoising
    denoised, sigma_map = complex_wishart_mppca(
        noisy_data, 
        mask=mask,
        patch_radius=2,  # 5x5x5 patches
        return_sigma=True,
        verbose=True
    )
    
    # Evaluate results
    print("\n📊 Denoising Results:")
    print(f"Mean estimated sigma: {np.mean(sigma_map[mask]):.4f}")
    print(f"True noise level: {noise_level:.4f}")
    
    # SNR improvement
    snr_results = compute_snr_improvement(signal, noisy_data, denoised, mask)
    print(f"\nSNR (noisy): {snr_results['snr_noisy']:.1f} dB")
    print(f"SNR (denoised): {snr_results['snr_denoised']:.1f} dB")
    print(f"SNR improvement: {snr_results['snr_gain']:.1f} dB")
    
    # Check edge preservation
    edge_signal = np.abs(np.gradient(signal[..., 0], axis=0)[mask])
    edge_denoised = np.abs(np.gradient(denoised[..., 0], axis=0)[mask])
    edge_preservation = np.corrcoef(edge_signal.flatten(), edge_denoised.flatten())[0, 1]
    print(f"\nEdge preservation correlation: {edge_preservation:.3f}")