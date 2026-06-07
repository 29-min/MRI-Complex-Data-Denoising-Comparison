
import napari
import scipy.io as sio
import numpy as np

def load_and_visualize():
    """Load MAT files and visualize in Napari"""
    
    # Load data
    print("Loading data...")
    original = sio.loadmat('napari_original_data.mat')
    noisy = sio.loadmat('napari_noisy_data.mat')
    denoised = sio.loadmat('napari_denoised_data.mat')
    comparison = sio.loadmat('napari_comparison_data.mat')
    
    # Extract data
    mag_orig = original['mag_orig']
    mag_noisy = noisy['mag_noisy']
    mag_denoised = denoised['mag_denoised']
    mask_brain = original['mask_brain']
    
    # Difference maps
    diff_noisy_orig = comparison['diff_noisy_orig']
    diff_denoised_orig = comparison['diff_denoised_orig']
    
    # Phase data
    phase_orig = original['phase_orig']
    phase_denoised = denoised['phase_denoised']
    phase_diff = comparison['phase_diff']
    
    print(f"Data shape: {mag_orig.shape}")
    
    # Create Napari viewer
    viewer = napari.Viewer(title="MP-PCA Denoising Results")
    
    # Add magnitude images
    viewer.add_image(mag_orig, name='Original Magnitude', colormap='gray')
    viewer.add_image(mag_noisy, name='Noisy Magnitude', colormap='gray')
    viewer.add_image(mag_denoised, name='Denoised Magnitude', colormap='gray')
    
    # Add difference maps
    viewer.add_image(diff_noisy_orig, name='Diff: Noisy - Original', colormap='bwr', 
                    contrast_limits=(-np.percentile(np.abs(diff_noisy_orig), 95), 
                                   np.percentile(np.abs(diff_noisy_orig), 95)))
    viewer.add_image(diff_denoised_orig, name='Diff: Denoised - Original', colormap='bwr',
                    contrast_limits=(-np.percentile(np.abs(diff_denoised_orig), 95), 
                                   np.percentile(np.abs(diff_denoised_orig), 95)))
    
    # Add phase images
    viewer.add_image(phase_orig, name='Original Phase', colormap='hsv')
    viewer.add_image(phase_denoised, name='Denoised Phase', colormap='hsv')
    viewer.add_image(phase_diff, name='Phase Difference', colormap='hot')
    
    # Add brain mask
    viewer.add_image(mask_brain, name='Brain Mask', colormap='gray', opacity=0.3)
    
    print("Napari viewer opened successfully!")
    print("Use the layer controls to navigate between echoes and slices.")
    print("Tip: Hold Shift and scroll to navigate through echoes (4th dimension)")
    
    return viewer

if __name__ == "__main__":
    viewer = load_and_visualize()
    napari.run()
