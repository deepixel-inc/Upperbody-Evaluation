# 🚀 Evaluation Code Usage Guide

**Portrait(Upperbody) segmentation 모델**의 성능 측정을 위한 평가 코드(`compute_seg_metrics.py`) 사용법 안내입니다.

---

## 목차
1. [Quick Start](#1-quick-start-)
2. [Arguments](#2-arguments-인자-상세-정보-)
3. [Data Format](#3-data-format-데이터-형식-가이드-)
4. [Directory Structure](#4-directory-structure--권장-디렉토리-구조)
5. [SDK Specification](#5-sdk-specification-)
---

## 1. Quick Start ⚡
`compute_seg_metrics.py` 안에 `compute_metrics` 함수를 호출하여 성능 평가를 수행할 수 있습니다.

```python
# 함수 호출 예시
miou, mean_bd_f1, mean_biou = compute_metrics(
    gt_paths=gt_image_list,  # ["path/to/00001.png", "path/to/00002.png", ... ]
    pred_mask_paths=prediction_list # ["path/to/00001.png", "path/to/00002.png", ... ]
)
```

| Metric | Name | Description |
| :--- | :--- | :--- |
| **`mIoU`** | Mean Intersection over Union | 정답 영역과 예측 영역의 중첩도를 측정하는 표준 지표입니다. |
| **`mean_bd_F1`** | Mean Boundary F1-score | 객체의 외곽선(Boundary)이 얼마나 정확한지 측정하며, 경계의 정밀도를 평가합니다. |
| **`mean_biou`** | Mean Boundary IoU | 경계 영역 근처의 IoU를 집중적으로 계산하여, 세밀한 분할 능력을 평가합니다. |

### 📊 **지표 해석 방법**
* 모든 지표는 **0에서 1 사이**의 값을 가집니다.
* **1에 가까울수록** 모델의 예측이 정답(GT)과 일치함을 의미합니다.
* 일반적으로 `mIoU`는 전체적인 영역을, `mean_bd_F1`과 `mean_biou`는 세부적인 윤곽선의 정확도를 파악할 때 유용합니다.


<br>

## 2. Arguments (인자 상세 정보) 📥

`compute_metrics` 함수는 평가 데이터셋의 경로 정보를 담은 두 개의 리스트를 인자로 받습니다.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`gt_paths`** | `List[str]` | 정답 데이터(Ground Truth)의 절대 혹은 상대 경로 리스트입니다. |
| **`pred_mask_paths`** | `List[str]` | 모델이 생성한 예측 마스크(Prediction)의 파일 경로 리스트입니다. |

> **⚠️ 주의사항: 데이터 매칭 확인**  
> * `gt_paths`와 `pred_mask_paths` 리스트 내의 파일 순서는 서로 일치해야 합니다. (예: `gt_paths[0]`의 정답 이미지는 `pred_mask_paths[0]`의 예측 이미지와 동일한 대상이어야 함)  
> 
> * **🏷️ Naming Tip:** 정답 이미지와 결과물의 파일명을 동일하게 설정하세요. (예: `00001.png` ↔ `00001.png` ; *데이터 예시 (Visual Example) 부분 확인*)

---

<br>

## 3. Data Format (데이터 형식 가이드) 📐

평가에 사용되는 모든 이미지는 **이진 마스크(Binary Mask)** 형태를 권장하며, 구체적인 픽셀 값 규정은 다음과 같습니다.

### 💎 **Pixel Value Specification**
정확한 평가지표 산출을 위해 이미지의 픽셀 값은 아래 기준을 따릅니다.

* **배경 (Background):** 반드시 **`0`**으로 채워져야 합니다.
* **전경 (Foreground):** **`0보다 큰 값`** (예: 1, 128, 200, 255 등)을 가질 수 있습니다. 
* **인식 로직:** 평가 알고리즘은 픽셀 값이 `0`이면 배경으로, `pixel > 0`이면 모두 유효한 객체(Foreground) 영역으로 간주합니다.

### 🎨 **데이터 예시 (Visual Example)**
이미지는 아래와 같이 배경과 객체가 명확히 분리된 마스크 형태여야 합니다.


| RGB Image | Ground Truth (*예시: 배경(0)과 전경(200)이 분리된 마스크 이미지*) |
| :---: | :---: |
| ![RGB Image Example](image/00001.png) | ![GT Image Example](ground_truth/EG1800_00001.png) 
| 00001.png | 00001.png |
<br>

### 🛠️ Preprocessing Note (전처리 유의사항)
* 4-Channel (BGRA): cv2.IMREAD_UNCHANGED로 로드시 4번째 채널 (Alpha)이 존재할 수 있습니다. 평가 전 반드시 1채널로 변환해야 합니다. 

---

<br>

## 4. Directory Structure 📂 (권장 디렉토리 구조)

평가 프로세스의 효율성과 파일 경로 리스트 생성을 용이하게 하기 위해 아래와 같은 디렉토리 구조를 권장합니다.

```text
project_root/
│
|── ground_truth/           # 정답 마스크 이미지 (배경: 0, 전경: > 0)
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
│── image/                  # RGB 이미지 (png or jpg 상관없음, 단순 예시용으로 평가 시 사용 안함)
|   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
│── predictions/            # 모델이 생성한 예측 결과물
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
│
└── compute_seg_metrics.py  # compute_metrics를 실행하는 메인 스크립트
```
---

<br>


## 5. SDK Specification 📋
<div style="border: 1px solid #e1e4e8; border-radius: 12px; padding: 20px; background-color: #0084ff00; line-height: 1.6;">
<div align="right">ver.202604 | DEEPIXEL</div>

## **DEEPIXEL VISION SDK**
### **>> Upperbody Segmentation SDK**

- Fast and accurate segmentation of the human upper body in real-time video streams.
- Robust algorithms against complex backgrounds, motion blurs, and light changes.
- Able to precisely extract and track upper body regions seamlessly in photos and video streaming.
- Applicable solutions across industries: Virtual backgrounds for video conferencing, portrait mode (background blur), AR try-on, live streaming, and photo editing.

<br>

### **TECHNICAL SPECIFICATION**
### **1. Performance Specification**
| Phone Model | Year of release | CPU Name | CPU Freq [MHz] | Cores | Tier | Target [ms] | Target CPU Usage [%] | Inference Time [ms] | CPU <br> Usage [%] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Samsung Galaxy S9 | 2018 | Exynos 9810 | 2700 | 8 | Android-Low | 20 | 10 | 7~8 (-13) | 2.75 (-7.25) |
| Samsung Galaxy A52s | 2021 | Snapdragon 778G | 2400 | 8 | Android-Mid | 9 | 4 | 8.15 (-0.85) | 3.04 (-0.96) |
| Samsung Galaxy Fold3 | 2021 | Snapdragon 888 | 2800 | 8 | Android-High | 5 | 4 | 4.99 (-0.01) | 1.86 (-2.14) |
| Samsung Galaxy S22 | 2022 | Snapdragon 8 Gen 1 | 3000 | 8 | Android-High | 5 | 4 | 4.25 (-0.75) | 1.58 (-2.42) |
| iPhone 6s | 2015 | A9 | 1840 | 2 | iOS-Low | 15 | 20 | 12~13 (-3) | 19.5~20.5 (-0.5) |
| iPhone XS | 2018 | A12 Bionic | 2490 | 6 | iOS-Mid | 7 | 4 | 5.55 (-1.45) | 3~4 (-1) |
| iPhone 14 Pro | 2022 | A16 Bionic | 3460 | 6 | iOS-High | 5 | 4 | 3.09 (-1.91) | 1.8 (-2.2) |

\* **GPU Usage: 0(%)** (GPU not used)

---
### **2. Model Specification**
#### **2.1. Accuracy Performance**
| Category | Metric | Value | Description |
| :--- | :--- | :--- | :--- |
| **Accuracy / Stability** | Mean IoU<br>Mean Boundary F1<br>Mean Boundary IoU | <div align="center">**85.23%**<br>**88.19%**<br>**80.93%**</div> | PP-HumanSeg14K dataset (2,398장) |
| **Accuracy / Stability** | Mean IoU<br>Mean Boundary F1<br>Mean Boundary IoU | <div align="center">**96.87%**<br>**97.19%**<br>**88.77%**</div> | EasyPortrait dataset (3,939장) |
| **Accuracy / Stability** | Mean IoU<br>Mean Boundary F1<br>Mean Boundary IoU | <div align="center">**95.67%**<br>**94.78%**<br>**83.23%**</div> | EG1800 dataset (1,707장) |


#### **2.2. SDK & Model Size**
| Category | Value | Uncompressed Details |
| :--- | :--- | :--- |
| <div align="center">**SDK Size**</div> | **Compressed**<br>└ 5.5 MB<br><br>**Uncompressed**<br>└ 13.9 MB | ├─ `sdk.so` [12.1 MB]<br>│&nbsp;&nbsp;&nbsp;&nbsp;*(includes opencv/tensorflowlite/c code)*<br>├─ `C++_shared.so` [1.0 MB]<br>└─ `model_file.tflite` [800 KB] |

---

### **3. Key Features**
- **Accuracy Value:** [0, 1] (Consistency between the predicted upper-body area and the ground truth.)
- **Segmentation Coverage:** Upper Body, Clothing, Shoulder & Arms, Background Exclusion
- **Multi-person:** Capable
- **Supported Platforms:** iOS, Android
