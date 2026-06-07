
import napari
import scipy.io as sio
import numpy as np

# Load denoised results
data = sio.loadmat('denoised_results.mat')

# Extract data
mag_denoised = data['mag_denoised']
mask_brain = data['mask_brain']

print(f"Loaded denoised data shape: {mag_denoised.shape}")

# Create Napari viewer
viewer = napari.Viewer(title="MP-PCA Denoised Results")

# Add denoised magnitude image
viewer.add_image(mag_denoised, name='Denoised Magnitude', colormap='gray')

# Add brain mask (optional)
viewer.add_image(mask_brain, name='Brain Mask', colormap='gray', opacity=0.3, visible=False)

print("Napari viewer opened!")
print("Tip: Use Shift + scroll to navigate through echoes (4th dimension)")

napari.run()
