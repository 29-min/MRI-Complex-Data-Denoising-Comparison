import numpy as np
import scipy.io as sio
import time
import pandas as pd
import matplotlib.pyplot as plt
from complex_localpca import mppca
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

# ----------------------- Parameters -----------------------
PATCH_RADIUS = 3        # patch radius for 7×7×7
EPSILON      = 1e-12    # to avoid divide-by-zero in SNR
SLICE_IDX    = None      # None이면 중앙 슬라이스 (Z//2) 사용

# ----------------------- Load data -----------------------
# Original complex data + brain mask
orig_mat    = sio.loadmat('meas_gre_dir1.mat')
meas_gre    = orig_mat['meas_gre']                # shape (X,Y,Z,Ne), complex
mask_brain  = orig_mat['mask_brain'].astype(bool) # shape (X,Y,Z)

# Noisy real & imag (noise case)
noise_mat   = sio.loadmat('noisy_meas_gre_dir1_10.mat')
noisy_real  = noise_mat['noisy_real'].astype(np.float32)
noisy_imag  = noise_mat['noisy_imag'].astype(np.float32)

# Dimensions
X, Y, Z, Ne = meas_gre.shape
print(f"Data dimensions: {X}×{Y}×{Z}×{Ne}")
print(f"Total number of slice-echo combinations: {Z * Ne}")

X, Y, Z, Ne = meas_gre.shape
if SLICE_IDX is None:
    SLICE_IDX = Z // 2

mag_orig  = np.abs(meas_gre)
mag_noisy = np.sqrt(noisy_real**2 + noisy_imag**2)

# ----------------------- Complex MP-PCA -----------------------
# build multi-channel array: real & imag concatenated
multi_ch = np.concatenate([noisy_real, noisy_imag], axis=3)  # (X,Y,Z,2*Ne)

print("▶ Running complex MP-PCA…")
t1 = time.time()
den_all = mppca(
    multi_ch,
    mask=mask_brain,
    patch_radius=PATCH_RADIUS,
    pca_method='svd',
    return_sigma=False
)
print(f"   Completed in {time.time()-t1:.1f}s")

real_den_cmplx = den_all[..., :Ne]
imag_den_cmplx = den_all[..., Ne:]
denoised = real_den_cmplx + 1j * imag_den_cmplx  # 복소수 복원
mag_den_cmplx  = np.abs(real_den_cmplx + 1j * imag_den_cmplx)         # (X,Y,Z,Ne)

# ----------------------- All Metric Functions -----------------------
def snr_diff_rician(ref, test, roi):
    """Bias-corrected SNR calculation"""
    s = []
    for c in range(ref.shape[3]):
        mu        = ref[..., c][roi].mean()
        sigma_raw = (test[..., c] - ref[..., c])[roi].std(ddof=1)
        s_corr    = np.sqrt(max(mu**2 - 2 * sigma_raw**2, 0.0))
        s.append(s_corr / (sigma_raw + EPSILON) * np.sqrt(2))
    return np.asarray(s)

def calculate_psnr(ref, test, roi):
    """PSNR calculation for each echo"""
    psnr_values = []
    for c in range(ref.shape[3]):
        ref_roi = ref[..., c][roi]
        test_roi = test[..., c][roi]
        
        mse = np.mean((ref_roi - test_roi) ** 2)
        if mse == 0:
            psnr_values.append(float('inf'))
        else:
            max_pixel = np.max(ref_roi)
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            psnr_values.append(psnr)
    return np.asarray(psnr_values)

def calculate_nrmse(ref, test, roi):
    """NRMSE calculation for each echo"""
    nrmse_values = []
    for c in range(ref.shape[3]):
        ref_roi = ref[..., c][roi]
        test_roi = test[..., c][roi]
        
        rmse = np.sqrt(np.mean((ref_roi - test_roi) ** 2))
        dynamic_range = np.max(ref_roi) - np.min(ref_roi)
        
        if dynamic_range == 0:
            nrmse_values.append(0)
        else:
            nrmse = rmse / dynamic_range
            nrmse_values.append(nrmse)
    return np.asarray(nrmse_values)

def calculate_hfen(ref, test, roi):
    """HFEN calculation for each echo"""
    # 라플라시안 커널 정의
    laplacian_kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
    
    hfen_values = []
    for c in range(ref.shape[3]):
        # 각 슬라이스별로 고주파 성분 추출 후 전체 HFEN 계산
        hf_errors = []
        for z in range(ref.shape[2]):
            if not roi[:, :, z].any():
                continue
                
            ref_slice = ref[:, :, z, c]
            test_slice = test[:, :, z, c]
            
            # 고주파 성분 추출
            hf_ref = ndimage.convolve(ref_slice, laplacian_kernel, mode='constant')
            hf_test = ndimage.convolve(test_slice, laplacian_kernel, mode='constant')
            
            # ROI 내에서만 계산
            hf_error = (hf_ref - hf_test)[roi[:, :, z]]
            hf_errors.extend(hf_error.flatten())
        
        if len(hf_errors) > 0:
            hfen = np.linalg.norm(hf_errors)
            hfen_values.append(hfen)
        else:
            hfen_values.append(0)
    
    return np.asarray(hfen_values)

# ----------------------- Echo-wise calculations -----------------------
# SNR (기존)
snr_noisy_bc = snr_diff_rician(mag_orig, mag_noisy, mask_brain)
snr_cmplx_bc = snr_diff_rician(mag_orig, mag_den_cmplx, mask_brain)

# PSNR
psnr_noisy = calculate_psnr(mag_orig, mag_noisy, mask_brain)
psnr_cmplx = calculate_psnr(mag_orig, mag_den_cmplx, mask_brain)

# NRMSE
nrmse_noisy = calculate_nrmse(mag_orig, mag_noisy, mask_brain)
nrmse_cmplx = calculate_nrmse(mag_orig, mag_den_cmplx, mask_brain)

# HFEN
hfen_noisy = calculate_hfen(mag_orig, mag_noisy, mask_brain)
hfen_cmplx = calculate_hfen(mag_orig, mag_den_cmplx, mask_brain)

# Echo-wise summary DataFrame
df_echo_all = pd.DataFrame({
    'Echo':           np.arange(Ne),
    'SNR_Noisy_BC':   snr_noisy_bc,
    'SNR_Complex_BC': snr_cmplx_bc,
    'ΔSNR_Complex':   snr_cmplx_bc - snr_noisy_bc,
    'PSNR_Noisy':     psnr_noisy,
    'PSNR_Complex':   psnr_cmplx,
    'ΔPSNR_Complex':  psnr_cmplx - psnr_noisy,
    'NRMSE_Noisy':    nrmse_noisy,
    'NRMSE_Complex':  nrmse_cmplx,
    'ΔNRMSE_Complex': nrmse_cmplx - nrmse_noisy,
    'HFEN_Noisy':     hfen_noisy,
    'HFEN_Complex':   hfen_cmplx,
    'ΔHFEN_Complex':  hfen_cmplx - hfen_noisy
}).round(4)

# 평균 행 추가
avg_row = {
    'Echo':           'Average',
    'SNR_Noisy_BC':   df_echo_all['SNR_Noisy_BC'].mean(),
    'SNR_Complex_BC': df_echo_all['SNR_Complex_BC'].mean(),
    'ΔSNR_Complex':   df_echo_all['ΔSNR_Complex'].mean(),
    'PSNR_Noisy':     df_echo_all['PSNR_Noisy'].mean(),
    'PSNR_Complex':   df_echo_all['PSNR_Complex'].mean(),
    'ΔPSNR_Complex':  df_echo_all['ΔPSNR_Complex'].mean(),
    'NRMSE_Noisy':    df_echo_all['NRMSE_Noisy'].mean(),
    'NRMSE_Complex':  df_echo_all['NRMSE_Complex'].mean(),
    'ΔNRMSE_Complex': df_echo_all['ΔNRMSE_Complex'].mean(),
    'HFEN_Noisy':     df_echo_all['HFEN_Noisy'].mean(),
    'HFEN_Complex':   df_echo_all['HFEN_Complex'].mean(),
    'ΔHFEN_Complex':  df_echo_all['ΔHFEN_Complex'].mean()
}
df_echo_all = pd.concat([df_echo_all, pd.DataFrame([avg_row])], ignore_index=True)

# ----------------------- Slice-wise calculations -----------------------
def snr_slice_df_bc(ref, test, roi, label):
    rows = []
    Xdim, Ydim, Zdim, C = ref.shape
    for z in range(Zdim):
        mask2d = roi[:, :, z]
        if not mask2d.any(): continue
        for c in range(C):
            mu        = ref[:, :, z, c][mask2d].mean()
            sigma_raw = (test[:, :, z, c] - ref[:, :, z, c])[mask2d].std(ddof=1)
            s_corr    = np.sqrt(max(mu**2 - 2 * sigma_raw**2, 0.0))
            snr_val   = s_corr / (sigma_raw + EPSILON) * np.sqrt(2)
            rows.append({'Slice': z, 'Echo': c, label: round(snr_val, 4)})
    return pd.DataFrame(rows)

def psnr_slice_df(ref, test, roi, label):
    rows = []
    X, Y, Z, C = ref.shape
    for z in range(Z):
        mask2d = roi[:, :, z]
        if not mask2d.any(): continue
        for c in range(C):
            ref_roi = ref[:, :, z, c][mask2d]
            test_roi = test[:, :, z, c][mask2d]
            
            mse = np.mean((ref_roi - test_roi) ** 2)
            if mse == 0:
                psnr_val = float('inf')
            else:
                max_pixel = np.max(ref_roi)
                psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            rows.append({'Slice': z, 'Echo': c, label: round(psnr_val, 4)})
    return pd.DataFrame(rows)

def nrmse_slice_df(ref, test, roi, label):
    rows = []
    X, Y, Z, C = ref.shape
    for z in range(Z):
        mask2d = roi[:, :, z]
        if not mask2d.any(): continue
        for c in range(C):
            ref_roi = ref[:, :, z, c][mask2d]
            test_roi = test[:, :, z, c][mask2d]
            
            rmse = np.sqrt(np.mean((ref_roi - test_roi) ** 2))
            dynamic_range = np.max(ref_roi) - np.min(ref_roi)
            
            if dynamic_range == 0:
                nrmse_val = 0
            else:
                nrmse_val = rmse / dynamic_range
            
            rows.append({'Slice': z, 'Echo': c, label: round(nrmse_val, 4)})
    return pd.DataFrame(rows)

def hfen_slice_df(ref, test, roi, label):
    laplacian_kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
    rows = []
    X, Y, Z, C = ref.shape
    for z in range(Z):
        mask2d = roi[:, :, z]
        if not mask2d.any(): continue
        for c in range(C):
            ref_slice = ref[:, :, z, c]
            test_slice = test[:, :, z, c]
            
            # 고주파 성분 추출
            hf_ref = ndimage.convolve(ref_slice, laplacian_kernel, mode='constant')
            hf_test = ndimage.convolve(test_slice, laplacian_kernel, mode='constant')
            
            # ROI 내에서만 계산
            hf_error = (hf_ref - hf_test)[mask2d]
            hfen_val = np.linalg.norm(hf_error)
            
            rows.append({'Slice': z, 'Echo': c, label: round(hfen_val, 4)})
    return pd.DataFrame(rows)

def ssim_slice_df(ref, test, roi, label):
    rows = []
    X, Y, Z, C = ref.shape
    for z in range(Z):
        if not roi[:, :, z].any(): continue
        for c in range(C):
            r2, t2 = ref[:, :, z, c], test[:, :, z, c]
            dr = r2.max() - r2.min()
            if dr == 0: continue
            val, _ = ssim(r2, t2, data_range=dr, full=True)
            rows.append({"Slice": z, "Echo": c, label: round(val, 4)})
    return pd.DataFrame(rows)

# 모든 slice-wise 메트릭 계산
print("▶ Calculating slice-wise metrics...")
df_snr_noisy = snr_slice_df_bc(mag_orig, mag_noisy, mask_brain, 'SNR_Noisy_BC')
df_snr_cmplx = snr_slice_df_bc(mag_orig, mag_den_cmplx, mask_brain, 'SNR_Complex_BC')

df_psnr_noisy = psnr_slice_df(mag_orig, mag_noisy, mask_brain, 'PSNR_Noisy')
df_psnr_cmplx = psnr_slice_df(mag_orig, mag_den_cmplx, mask_brain, 'PSNR_Complex')

df_nrmse_noisy = nrmse_slice_df(mag_orig, mag_noisy, mask_brain, 'NRMSE_Noisy')
df_nrmse_cmplx = nrmse_slice_df(mag_orig, mag_den_cmplx, mask_brain, 'NRMSE_Complex')

df_hfen_noisy = hfen_slice_df(mag_orig, mag_noisy, mask_brain, 'HFEN_Noisy')
df_hfen_cmplx = hfen_slice_df(mag_orig, mag_den_cmplx, mask_brain, 'HFEN_Complex')

df_ssim_slice = ssim_slice_df(mag_orig, mag_den_cmplx, mask_brain, 'SSIM')

# 모든 slice-wise 데이터 합치기
df_slice_all = (
    df_snr_noisy
    .merge(df_snr_cmplx, on=['Slice', 'Echo'])
    .merge(df_psnr_noisy, on=['Slice', 'Echo'])
    .merge(df_psnr_cmplx, on=['Slice', 'Echo'])
    .merge(df_nrmse_noisy, on=['Slice', 'Echo'])
    .merge(df_nrmse_cmplx, on=['Slice', 'Echo'])
    .merge(df_hfen_noisy, on=['Slice', 'Echo'])
    .merge(df_hfen_cmplx, on=['Slice', 'Echo'])
    .merge(df_ssim_slice, on=['Slice', 'Echo'])
    .sort_values(['Echo', 'Slice'], ignore_index=True)
)

# 차이값 계산
df_slice_all['ΔSNR_Complex'] = df_slice_all['SNR_Complex_BC'] - df_slice_all['SNR_Noisy_BC']
df_slice_all['ΔPSNR_Complex'] = df_slice_all['PSNR_Complex'] - df_slice_all['PSNR_Noisy']
df_slice_all['ΔNRMSE_Complex'] = df_slice_all['NRMSE_Complex'] - df_slice_all['NRMSE_Noisy']
df_slice_all['ΔHFEN_Complex'] = df_slice_all['HFEN_Complex'] - df_slice_all['HFEN_Noisy']

# Echo별 SSIM 요약
df_ssim_echo = (
    df_ssim_slice
    .groupby('Echo')['SSIM']
    .mean()
    .reset_index()
    .round(4)
)
# 평균 행 추가
avg_ssim = {
    'Echo': 'Average',
    'SSIM': df_ssim_echo['SSIM'].mean()
}
df_ssim_echo = pd.concat([df_ssim_echo, pd.DataFrame([avg_ssim])], ignore_index=True)

# ----------------------- Save to Excel -----------------------
excel_filename = 'complete_mri_metrics_results.xlsx'
print(f'▶ Saving results to {excel_filename}…')

with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
    # Echo-wise 요약 (모든 메트릭)
    df_echo_all.to_excel(writer, sheet_name='Echo_Summary_All_Metrics', index=False)
    
    # Slice-wise 상세 (모든 메트릭)
    df_slice_all.to_excel(writer, sheet_name='Slice_by_Echo_All_Metrics', index=False)
    
    # 개별 메트릭 시트들 (기존 호환성 유지)
    df_ssim_slice.to_excel(writer, sheet_name='Slice_by_Echo_SSIM', index=False)
    df_ssim_echo.to_excel(writer, sheet_name='Echo_Summary_SSIM', index=False)

# ----------------------- Phase analysis -----------------------
phase_orig = np.angle(meas_gre)
phase_denoised = np.angle(denoised)
phase_diff = np.abs(phase_orig - phase_denoised)
print(f"Mean phase difference: {np.mean(phase_diff[mask_brain]):.4f} rad")

print('✔ All metrics calculated and results saved successfully!')

# ----------------------- Results Summary Print -----------------------
print("\n" + "="*80)
print("📊 ECHO-WISE METRICS SUMMARY")
print("="*80)
print(df_echo_all.to_string(index=False))

print(f"\n📁 Detailed results saved to: {excel_filename}")
print("   - Echo_Summary_All_Metrics: Echo별 모든 메트릭 요약")
print("   - Slice_by_Echo_All_Metrics: Slice별 상세 메트릭")
print("   - Slice_by_Echo_SSIM: SSIM 상세 데이터")
print("   - Echo_Summary_SSIM: SSIM 요약")

# ----------------------- Visualization -----------------------
vmin, vmax = np.percentile(mag_orig[mask_brain], (1, 99))

print("\n⋯ Echo-wise 5-column 시각화 생성")
n_echoes = mag_orig.shape[-1]
diff_vmax = np.percentile(
    np.abs(mag_noisy - mag_orig)[mask_brain], 99)

for e in range(Ne):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle(f"Noise Level 10 | Slice {SLICE_IDX} | Echo {e}", fontsize=20)

    axes[0].imshow(mag_noisy[:, :, SLICE_IDX, e], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Magnitude')
    axes[0].axis('off')

    axes[1].imshow(mag_den_cmplx[:, :, SLICE_IDX, e], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised_Complex')
    axes[1].axis('off')

    axes[2].imshow(mask_brain[:, :, SLICE_IDX], cmap='gray')
    axes[2].set_title('Mask Region')
    axes[2].axis('off')

    diff = mag_den_cmplx[:, :, SLICE_IDX, e] - mag_orig[:, :, SLICE_IDX, e]
    axes[3].imshow(diff, cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[3].set_title('Diff(Denoised - Orig)')
    axes[3].axis('off')

    diff = mag_noisy[:, :, SLICE_IDX, e] - mag_orig[:, :, SLICE_IDX, e]
    axes[4].imshow(diff, cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[4].set_title('Diff(noisy - Orig)')
    axes[4].axis('off')

    plt.subplots_adjust(wspace=0.03, hspace=0.1, top=0.9)
    plt.tight_layout()
    plt.show()

print("\n🎉 All analysis completed successfully!")