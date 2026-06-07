import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings

def complex_eig_mppca_dipy(data, mask=None, patch_radius=2, 
                          tau_factor=None, return_sigma=False,
                          phase_correct=True, verbose=True):
    """
    Complex eigenvalue decomposition MP-PCA using DIPY's threshold approach.
    
    DIPY threshold calculation:
    - For MP-PCA: tau_factor is automatically calculated if None
    - For LocalPCA: tau_factor = 2.3 (default)
    - Threshold: tau = tau_factor * sigma * sqrt((1 + sqrt(gamma))^2)
    
    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N_echoes)
        Complex-valued 4D data
    mask : ndarray, optional
        Brain mask
    patch_radius : int, optional
        Patch radius (default: 2 for 5x5x5)
    tau_factor : float, optional
        If None, automatically calculated using MP theory (DIPY approach)
        Default for LocalPCA is 2.3
    return_sigma : bool, optional
        Return noise map
    phase_correct : bool, optional
        Apply phase correction
    verbose : bool, optional
        Print progress
        
    Returns
    -------
    denoised : ndarray
        Denoised complex data
    sigma_map : ndarray (optional)
        Noise standard deviation map
    """
    if verbose:
        print("▶ Complex Eigenvalue MP-PCA (DIPY-style threshold)")
    
    # Validate input
    if not np.iscomplexobj(data):
        raise ValueError("Input must be complex-valued")
    
    X, Y, Z, N = data.shape
    
    # Create mask if needed
    if mask is None:
        mask = np.ones((X, Y, Z), dtype=bool)
    
    # Phase correction
    if phase_correct:
        data = _apply_phase_correction(data, mask, verbose)
    
    # Initialize outputs
    denoised = np.zeros_like(data)
    weights = np.zeros((X, Y, Z), dtype=np.float32)
    
    if return_sigma:
        sigma_map = np.zeros((X, Y, Z), dtype=np.float32)
    
    # Patch parameters
    pr = patch_radius
    patch_size = 2*pr + 1
    M = patch_size**3  # number of voxels
    
    if verbose:
        print(f"   Patch size: {patch_size}×{patch_size}×{patch_size} = {M} voxels")
        print(f"   Measurements: {N} echoes (complex)")
        print(f"   Aspect ratio γ = N/M = {N/M:.3f}")
    
    # Check MP validity
    if M < N:
        warnings.warn(f"M ({M}) < N ({N}): MP theory assumptions violated. "
                     f"Consider larger patch_radius.")
    
    # Process all voxels
    total_voxels = np.sum(mask)
    n_denoised = 0  # Track actual denoising
    
    with tqdm(total=total_voxels, desc="Denoising", disable=not verbose) as pbar:
        for k in range(pr, Z-pr):
            for j in range(pr, Y-pr):
                for i in range(pr, X-pr):
                    if not mask[i, j, k]:
                        continue
                    
                    # Extract patch
                    patch = data[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1]
                    
                    # Reshape to matrix (M voxels × N echoes)
                    X_patch = patch.reshape(M, N)
                    
                    # Denoise using complex eigenvalue decomposition
                    X_denoised, sigma, actually_denoised = _denoise_patch_complex_eig(
                        X_patch, M, N, tau_factor
                    )
                    
                    if actually_denoised:
                        n_denoised += 1
                    
                    # Reshape back
                    patch_denoised = X_denoised.reshape(
                        patch_size, patch_size, patch_size, N
                    )
                    
                    # Accumulate
                    denoised[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += patch_denoised
                    weights[i-pr:i+pr+1, j-pr:j+pr+1, k-pr:k+pr+1] += 1.0
                    
                    if return_sigma:
                        sigma_map[i, j, k] = sigma
                    
                    pbar.update(1)
    
    # Normalize
    weights[weights == 0] = 1
    denoised = denoised / weights[..., None]
    
    # Apply mask
    denoised[~mask] = 0
    
    if verbose:
        denoising_rate = 100 * n_denoised / total_voxels
        print(f"\n   Denoising rate: {denoising_rate:.1f}% of voxels")
        if denoising_rate < 50:
            print("   ⚠️  Low denoising rate. Consider adjusting tau_factor.")
    
    if return_sigma:
        # Smooth sigma map (DIPY style)
        sigma_map = gaussian_filter(sigma_map, sigma=1.0)
        return denoised, sigma_map
    
    return denoised


def _denoise_patch_complex_eig(X, M, N, tau_factor=None):
    """
    Denoise patch using complex eigenvalue decomposition with DIPY-style threshold.
    
    DIPY's approach:
    1. Compute eigenvalues of covariance matrix
    2. Estimate sigma from eigenvalue distribution
    3. If tau_factor=None: compute automatically using MP theory
    4. Threshold: keep eigenvalues > tau_factor * sigma^2 * (1+sqrt(gamma))^2
    """
    # Center the data
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean
    
    # Complex covariance matrix C = (1/M) * X^H * X
    C = (X_centered.conj().T @ X_centered) / M
    
    # Eigendecomposition of Hermitian matrix
    # All eigenvalues are real, eigenvectors are complex
    eigenvalues, eigenvectors = eigh(C)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Remove numerical noise (negative eigenvalues)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Estimate noise standard deviation
    sigma = _estimate_sigma_dipy_style(eigenvalues, M, N)
    
    # Compute threshold using DIPY approach
    if tau_factor is None:
        # Automatic tau_factor selection (MP-PCA mode)
        tau_factor = _compute_tau_factor_mp(eigenvalues, M, N, sigma)
    
    # MP distribution parameters
    gamma = N / M
    lambda_plus = (1 + np.sqrt(gamma))**2
    
    # DIPY threshold: tau = tau_factor * sigma^2 * lambda_plus
    tau = tau_factor * sigma**2 * lambda_plus
    
    # Select signal components
    n_components = np.sum(eigenvalues > tau)
    
    # Check if any denoising occurs
    actually_denoised = (n_components < N)
    
    if n_components == 0:
        # Keep at least one component
        n_components = 1
    elif n_components == N:
        # No denoising - all components kept
        # This often happens when tau_factor is too low
        pass
    
    # Reconstruct using selected components
    if n_components < N:
        # Project onto signal subspace
        U_signal = eigenvectors[:, :n_components]
        # Reconstruct: X_proj = X * U * U^H
        X_proj = X_centered @ U_signal @ U_signal.conj().T
        X_denoised = X_proj + X_mean
    else:
        # No denoising
        X_denoised = X
    
    return X_denoised, sigma, actually_denoised


def _estimate_sigma_dipy_style(eigenvalues, M, N):
    """
    Estimate noise sigma following DIPY's MP-PCA approach.
    
    DIPY uses the eigenvalue distribution and MP theory to estimate sigma.
    The approach is to find eigenvalues in the "bulk" of the MP distribution.
    """
    # Parameters
    gamma = N / M
    
    # Remove zero eigenvalues
    nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
    
    if len(nonzero_eigs) == 0:
        return 0.0
    
    # Initial estimate using smallest eigenvalues
    # DIPY uses bottom portion for initial estimate
    n_noise = max(1, int(0.1 * len(nonzero_eigs)))
    noise_eigs = np.sort(nonzero_eigs)[:n_noise]
    
    # Initial sigma estimate
    sigma_init = np.sqrt(np.mean(noise_eigs))
    
    # Find eigenvalues in MP bulk
    # Upper bound of MP distribution
    lambda_plus_est = sigma_init**2 * (1 + np.sqrt(gamma))**2
    
    # Find eigenvalues below theoretical upper bound (with margin)
    bulk_mask = nonzero_eigs <= (lambda_plus_est * 1.02)
    bulk_eigs = nonzero_eigs[bulk_mask]
    
    if len(bulk_eigs) > n_noise:
        # Refined estimate from bulk
        sigma = np.sqrt(np.mean(np.sort(bulk_eigs)[:n_noise]))
    else:
        sigma = sigma_init
    
    # Apply finite sample correction
    if gamma < 1:
        # MP theory correction for finite samples
        correction = 1 / (1 - np.sqrt(gamma))**2
        sigma = sigma * np.sqrt(correction)
    
    return sigma


def _compute_tau_factor_mp(eigenvalues, M, N, sigma):
    """
    Automatically compute tau_factor using MP theory (DIPY MP-PCA mode).
    
    When tau_factor=None in DIPY, it automatically selects the threshold
    based on the MP distribution properties.
    """
    gamma = N / M
    
    # Expected bulk edge of MP distribution
    lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
    
    # Find gap in eigenvalue distribution
    # This identifies where signal eigenvalues separate from noise
    eig_sorted = np.sort(eigenvalues)[::-1]
    
    # Compute eigenvalue gaps
    if len(eig_sorted) > 1:
        gaps = eig_sorted[:-1] - eig_sorted[1:]
        # Normalize gaps
        gaps_normalized = gaps / (eig_sorted[:-1] + 1e-10)
        
        # Find significant gap near expected threshold
        expected_idx = np.sum(eigenvalues > lambda_plus)
        search_range = max(1, int(0.2 * len(eigenvalues)))
        
        if expected_idx > 0 and expected_idx < len(gaps):
            # Look for largest gap near expected threshold
            start_idx = max(0, expected_idx - search_range)
            end_idx = min(len(gaps), expected_idx + search_range)
            
            gap_region = gaps_normalized[start_idx:end_idx]
            if len(gap_region) > 0:
                max_gap_idx = np.argmax(gap_region) + start_idx
                
                # Set threshold between eigenvalues at gap
                tau_eigenvalue = (eig_sorted[max_gap_idx] + 
                                 eig_sorted[max_gap_idx + 1]) / 2
                
                # Convert to tau_factor
                tau_factor = tau_eigenvalue / (sigma**2 * lambda_plus)
                
                # Ensure reasonable range
                tau_factor = np.clip(tau_factor, 0.8, 3.0)
            else:
                tau_factor = 1.0
        else:
            tau_factor = 1.0
    else:
        tau_factor = 1.0
    
    # Adjust based on matrix dimensions
    if M < 50:
        tau_factor *= 1.2  # More conservative for small patches
    elif M > 200:
        tau_factor *= 0.9  # Can be more aggressive for large patches
    
    return tau_factor


def _apply_phase_correction(data, mask, verbose):
    """
    Apply phase correction to improve denoising effectiveness.
    """
    if verbose:
        print("   Applying phase correction...")
    
    data_corrected = data.copy()
    
    for echo in range(data.shape[-1]):
        # Extract phase
        phase = np.angle(data[..., echo])
        
        # Smooth phase (only in mask)
        phase_smooth = phase.copy()
        phase_smooth[~mask] = 0
        phase_smooth = gaussian_filter(phase_smooth, sigma=3.0)
        
        # Remove smooth phase variations
        magnitude = np.abs(data[..., echo])
        phase_corrected = phase - phase_smooth
        
        # Reconstruct
        data_corrected[..., echo] = magnitude * np.exp(1j * phase_corrected)
    
    return data_corrected


# Utility functions for testing and evaluation
def test_complex_eig_denoising():
    """
    Test the complex eigenvalue MP-PCA implementation.
    """
    print("Testing Complex Eigenvalue MP-PCA with DIPY Threshold\n")
    
    # Create test data
    shape = (64, 64, 32, 6)  # 6 echoes
    
    # Brain mask
    center = [s//2 for s in shape[:3]]
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = ((x-center[0])**2/25**2 + (y-center[1])**2/25**2 + 
            (z-center[2])**2/15**2) < 1
    
    # Complex signal with T2* decay
    TE = np.array([5, 10, 15, 20, 25, 30]) * 1e-3
    T2_star = 25e-3
    
    signal = np.zeros(shape, dtype=np.complex128)
    for i in range(shape[3]):
        mag = mask.astype(float) * np.exp(-TE[i] / T2_star)
        phase = np.random.randn(*shape[:3]) * 0.2
        signal[..., i] = mag * np.exp(1j * phase)
    
    # Add complex noise
    noise_level = 0.05
    noise = noise_level * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    noisy = signal + noise
    
    print("Test 1: Automatic tau_factor (MP-PCA mode)")
    denoised1, sigma1 = complex_eig_mppca_dipy(
        noisy, mask=mask,
        patch_radius=2,
        tau_factor=None,  # Automatic
        return_sigma=True,
        verbose=True
    )
    
    print("\nTest 2: Fixed tau_factor=2.3 (LocalPCA mode)")
    denoised2, sigma2 = complex_eig_mppca_dipy(
        noisy, mask=mask,
        patch_radius=2,
        tau_factor=2.3,  # DIPY LocalPCA default
        return_sigma=True,
        verbose=True
    )
    
    print("\nTest 3: Lower tau_factor=1.5 (more aggressive)")
    denoised3, sigma3 = complex_eig_mppca_dipy(
        noisy, mask=mask,
        patch_radius=2,
        tau_factor=1.5,
        return_sigma=True,
        verbose=True
    )
    
    # Evaluate results
    print("\n📊 Results:")
    
    for i, (den, sig, name) in enumerate([
        (denoised1, sigma1, "Auto tau"),
        (denoised2, sigma2, "tau=2.3"),
        (denoised3, sigma3, "tau=1.5")
    ]):
        # Calculate metrics
        residual = noisy - den
        noise_power = np.mean(np.abs(residual[mask])**2)
        original_noise_power = np.mean(np.abs(noise[mask])**2)
        
        reduction = 100 * (1 - noise_power / original_noise_power)
        mean_sigma = np.mean(sig[mask])
        
        print(f"\n{name}:")
        print(f"  Noise reduction: {reduction:.1f}%")
        print(f"  Estimated σ: {mean_sigma:.4f} (true: {noise_level:.4f})")
        
        # Check if actually denoised
        diff = np.mean(np.abs(noisy[mask] - den[mask]))
        print(f"  Mean difference: {diff:.6f}")


if __name__ == "__main__":
    test_complex_eig_denoising()