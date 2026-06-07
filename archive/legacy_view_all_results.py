
import napari
import scipy.io as sio
import numpy as np

def load_and_view_results():
    """Load saved results and create Napari viewer"""
    
    # Load data
    print("Loading saved results...")
    data = sio.loadmat('all_denoising_results.mat')
    
    # Echo-averaged 3D data for clean visualization
    mag_orig_avg = data['mag_orig_avg']
    mag_noisy_avg = data['mag_noisy_avg']
    deno_mag_avg = data['deno_mag_avg']
    deno_split_avg = data['deno_split_avg']
    deno_cmplx_avg = data['deno_cmplx_avg']
    mask_brain = data['mask_brain']
    
    print(f"Echo-averaged data shape: {mag_orig_avg.shape}")
    print(f"Original 4D data also available in MAT file")
    
    # Create viewer
    viewer = napari.Viewer(title="MP-PCA Denoising Results (Echo-averaged)")
    
    # Set contrast limits
    vmin, vmax = np.percentile(mag_orig_avg[mask_brain.astype(bool)], [1, 99])
    
    # Add echo-averaged images
    viewer.add_image(mag_orig_avg, name='1. Original', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(mag_noisy_avg, name='2. Noisy', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(deno_mag_avg, name='3. Magnitude MP-PCA', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(deno_split_avg, name='4. Split MP-PCA', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(deno_cmplx_avg, name='5. Complex MP-PCA', colormap='gray', contrast_limits=[vmin, vmax])
    
    # Add mask
    viewer.add_image(mask_brain, name='Brain Mask', colormap='red', opacity=0.3, visible=False)
    
    # Add difference maps
    diff_vmax = np.percentile(np.abs(mag_noisy_avg - mag_orig_avg), 95)
    viewer.add_image(deno_mag_avg - mag_orig_avg, name='Diff: Mag - Orig', 
                    colormap='bwr', contrast_limits=[-diff_vmax, diff_vmax], visible=False)
    viewer.add_image(deno_split_avg - mag_orig_avg, name='Diff: Split - Orig', 
                    colormap='bwr', contrast_limits=[-diff_vmax, diff_vmax], visible=False)
    viewer.add_image(deno_cmplx_avg - mag_orig_avg, name='Diff: Complex - Orig', 
                    colormap='bwr', contrast_limits=[-diff_vmax, diff_vmax], visible=False)
    
    print("Napari viewer ready!")
    print("Navigate through Z-slices with mouse wheel")
    return viewer

def load_4d_data():
    """Load original 4D data for detailed analysis"""
    print("Loading 4D data...")
    data = sio.loadmat('all_denoising_results.mat')
    
    # Create viewer for 4D data
    viewer = napari.Viewer(title="MP-PCA Denoising Results (4D - All Echoes)")
    
    # Original 4D data
    mag_orig_4d = data['mag_orig_4d']
    mag_noisy_4d = data['mag_noisy_4d']
    deno_mag_4d = data['deno_mag_4d']
    deno_split_4d = data['deno_split_4d']
    deno_cmplx_4d = data['deno_cmplx_4d']
    mask_brain = data['mask_brain']
    
    vmin, vmax = np.percentile(mag_orig_4d[mask_brain.astype(bool)], [1, 99])
    
    viewer.add_image(mag_orig_4d, name='1. Original (4D)', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(mag_noisy_4d, name='2. Noisy (4D)', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(deno_mag_4d, name='3. Magnitude MP-PCA (4D)', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(deno_split_4d, name='4. Split MP-PCA (4D)', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(deno_cmplx_4d, name='5. Complex MP-PCA (4D)', colormap='gray', contrast_limits=[vmin, vmax])
    
    print("4D Napari viewer ready!")
    print("Use Shift+scroll to navigate through echoes")
    return viewer

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "4d":
        viewer = load_4d_data()
    else:
        viewer = load_and_view_results()
    
    napari.run()
