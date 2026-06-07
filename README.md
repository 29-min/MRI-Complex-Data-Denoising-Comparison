# MRI Complex Data Denoising Comparison

이 폴더는 `최종연구보고서.docx.pdf`를 기준으로 연구 내용을 파악한 뒤, 연구에 사용된 코드만 따로 복사해 정리한 것이다. `.xlsx`, `.mat`, `.npy`, 이미지, PDF 같은 데이터/결과 파일은 포함하지 않았다.

## 연구 개요

본 연구의 주제는 multi-echo 복소수 MRI 데이터의 노이즈 제거 성능 비교이다. MRI 원본 신호는 `real + i * imaginary` 형태의 복소수 데이터이며, 보고서에서는 `(256, 224, 176, 6)` 구조의 6 echo MRI 데이터를 사용했다. 원본 변수명과 보고서 본문에는 `meas_gre`, `Gradient Echo 기반 MRI(GRE MRI)`라는 표현이 등장하지만, 이 저장소의 핵심은 특정 시퀀스 자체보다 복소수 MRI 데이터에서 magnitude, real/imaginary split, complex-concatenated MP-PCA 접근을 비교한 과정이다.

임상에서 흔히 사용하는 magnitude 영상은 phase 정보를 잃고 Gaussian 노이즈가 Rician 노이즈로 바뀌는 문제가 있으므로, 본 연구는 real/imaginary 정보를 보존한 MP-PCA 기반 denoising 방법을 비교했다.

주요 실험 흐름은 다음과 같다.

1. Real/imaginary 성분에 10-50% 수준의 Gaussian 노이즈 추가
2. Magnitude 도메인 MP-PCA denoising
3. Real/imaginary 분리 MP-PCA denoising
4. Real/imaginary를 하나의 실수 행렬로 연결한 복소수-실수 행렬 MP-PCA denoising
5. Complex Wishart / Hermitian 기반 복소수 MP-PCA 접근 실험
6. SNR, SSIM 등 정량 지표와 diff map 기반 시각화로 성능 비교

보고서 결론상 모든 방식이 노이즈 제거 효과를 보였고, 복소수-실수 행렬 방식은 여러 echo의 상관 정보를 함께 활용해 일부 구간에서 magnitude 방식보다 좋은 성능을 보였다. 다만 echo가 증가하거나 신호가 약해지는 구간에서는 magnitude 방식이 더 안정적인 경우가 있었고, real/imaginary 분리 방식은 두 성분의 상관관계를 충분히 활용하지 못해 상대적으로 성능이 낮게 나타났다.

## 사용 기술

- Python 기반 수치 계산: `numpy`, `scipy`
- MRI 데이터 입출력: `scipy.io.loadmat`, `scipy.io.savemat`
- MP-PCA denoising: DIPY `localpca`/`mppca` 계열 구현 수정 및 활용
- 선형대수: PCA, SVD, eigenvalue decomposition, Hermitian covariance
- 랜덤 행렬 이론: Marcenko-Pastur 분포 기반 신호/노이즈 고유값 분리
- 복소수 MRI 처리: real/imaginary 분리, real/imaginary concatenation, phase 보존
- 노이즈 모델링: Gaussian noise, Rician noise, Rician bias correction
- 평가 지표: SNR, SSIM, PSNR, NRMSE, HFEN
- 시각화: `matplotlib`, `napari`
- 결과 표 생성: `pandas`, `xlsxwriter`

## 폴더 구조

```text
README.md
requirements.txt
final_code/
  modules/        # 최종 실험에서 사용된 denoising 알고리즘 모듈
  pipelines/      # 데이터 로드, 실행, 평가, 시각화 스크립트
  notebooks/      # 최종 실험 및 비교에 사용된 노트북
reference_code/   # 루트에 있던 관련 참고 코드와 초기 실험 노트북
```

## 주요 코드 설명

- `final_code/notebooks/종합.ipynb`: 최종 비교 실험 노트북. 복소수-실수 행렬 방식 denoising, SNR/SSIM 등 지표 계산, 결과 저장 및 시각화 코드를 포함한다.
- `final_code/pipelines/new.py`: `종합.ipynb`와 유사한 실행형 스크립트. `complex_localpca.mppca`를 사용해 real/imaginary concatenation 방식으로 denoising하고 SNR, SSIM, PSNR, NRMSE, HFEN을 계산한다.
- `final_code/modules/complex_localpca.py`: DIPY localpca 기반 구현을 복사/수정한 핵심 모듈. 실수부/허수부의 음수 값을 보존하기 위해 magnitude 전용 clipping을 제거한 MP-PCA 구현이다.
- `final_code/modules/complex_localpca_ii.py`, `complex_iii.py`, `complex_xi.py`: complex/local PCA 실험 변형. SVD/eigenvalue 기반 처리와 threshold 조정 실험이 포함되어 있다.
- `final_code/modules/twopcaii.py`, `twopcaiii.py`, `fpii.py`: 복소수 신호를 real/imaginary 2D vector 또는 concatenated feature로 변환하여 MP-PCA를 적용하는 실험 코드다.
- `final_code/modules/lastcomplex.py`, `cw.py`, `cw2.py`, `cw3.py`, `cw4.py`: Complex Wishart, Hermitian covariance, 복소수 SVD/eigenvalue 기반 denoising 접근을 실험한 코드다.
- `final_code/modules/fpreservation.py`: denoising 후 구조/경계 정보를 보존하기 위한 후처리 실험 코드다.
- `final_code/pipelines/testpcaiii.py`: complex-as-2D-vector MP-PCA 접근의 테스트/검증용 스크립트다.
- `final_code/pipelines/view_denoised.py`, `visualize_napari.py`: denoising 결과를 Napari에서 확인하기 위한 시각화 스크립트다.
- `reference_code/`: 루트 경로에 있던 관련 초기 실험 코드와 노트북을 별도로 보관했다.

## 실행 참고

코드 실행에는 원본 데이터 파일이 필요하지만, 이 정리 폴더에는 데이터 파일을 포함하지 않았다. 원본 스크립트는 보통 현재 작업 디렉터리에 다음 파일이 있다고 가정한다.

- `meas_gre_dir1.mat`
- `noisy_meas_gre_dir1_10.mat`
- `noisy_meas_gre_dir1_20.mat`
- `noisy_meas_gre_dir1_30.mat`
- `noisy_meas_gre_dir1_40.mat`
- `noisy_meas_gre_dir1_50.mat`

정리된 구조에서 실행하려면 모듈 경로를 함께 지정해야 한다.

```bash
PYTHONPATH=final_code/modules python final_code/pipelines/new.py
```

위 명령은 데이터 파일이 있는 디렉터리에서 실행하는 것을 전제로 한다. 일부 스크립트는 실행 시 `.xlsx` 또는 `.mat` 결과 파일을 새로 생성한다.

## 정리 기준

- 포함: `.py`, `.ipynb` 코드 파일
- 제외: `.xlsx`, `.mat`, `.npy`, `.png`, `.pdf`, `.DS_Store`, Python cache
- 최종 코드 기준: `마지막` 폴더
- 참고 코드 기준: 루트에 있던 관련 스크립트/노트북
