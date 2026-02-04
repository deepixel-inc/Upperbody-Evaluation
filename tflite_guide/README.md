# 🚀 TFLite 모델 벤치마크 테스트 가이드

#### 이 문서는 TensorFlow Lite(TFLite) 모델의 성능(추론 시간, 리소스 사용량 등)을 정량적으로 측정하기 위한 **TFLite Benchmark Tool** 사용법을 설명합니다.

| Benchmark 결과화면 | 주요 지표 설명 |
| :--- | :--- |
| <img src="sample/Deepixel Tflite Benchmark_main1.jpg" width="500"> <br> **[그림 1] TFLite 성능 측정 메인 결과** | <br>**⚙️실행 설정 [Current Settings]** <br> 벤치마크 수행 시 적용된 환경 설정값입니다. <br> 모델 성능을 측정하기 전, CPU 리소스 할당 및 하드웨어 가속 옵션을 정의하려면 <br> 우측 상단 ⚙️를 눌러주세요.` [그림2] 참조` <br> • **Threads (1)**: 추론 시 활용된 CPU 코어 수 <br> • **Target FPS (30.0)**: 목표 초당 프레임 수 <br> • **Warm-up Runs (10)**: 측정 전 안정화 실행 횟수 <br> • **Benchmark Runs (50)**: 실제 통계 데이터를 산출하기 위해 반복 실행한 횟수입니다. <br><br> **📊벤치마크 요약 [Benchmark Summary]** <br> • **Model Size:** 기기에 저장된 모델 파일의 물리적 용량 <br> • **Inference Time**: 1회 추론 평균 시간 <br> • **Actual FPS**: 실제 측정된 초당 프레임 수 <br> • **CPU Usage:** 모델 실행 중 CPU에 가해지는 연산 부하율 <br> • **Memory Usage**: 점유 중인 RAM 용량 <br> • **Model Name:** 테스트에 사용된 .tflite 모델 파일명 <br> • **Actual FPS:** 실제 구동 시 측정된 초당 프레임 수 <br> • **CPU Freq:** 벤치마크 중 CPU가 최대 클럭 성능을 얼마나 유지했는지 <br>
---


| Benchmark 결과화면 | 주요 지표 설명 |
| :--- | :--- |
| <img src="sample/Deepixel Tflite Benchmark_main3.jpg" width="500"> <br> **[그림 2] Benchmark Setting 수정 가능 화면** | **🛠️ 3. 벤치마크 설정 (Benchmark Setting)** <br> **`1. 기본 실행 설정`** <br> 모델이 구동되는 물리적 환경을 설정합니다. <br> • **Number of Threads (1):** 추론(inference) 프로세스에 할당할 CPU 코어의 개수 <br> • **Target FPS (30.0):** 앱이 목표로 하는 초당 프레임 수 <br> • **Warm-up Runs (10):** 본격적인 측정 전, 캐시 로드 및 하드웨어 가속기(Delegate) 초기화를 위해 사전 실행하는 횟수 <br> • **Benchmark Runs (50):** 통계적 신뢰도를 얻기 위해 반복 수행할 총 횟수 <br><br> **`2. 가속 옵션 (Acceleration Options)`** <br> 모바일 기기의 특수 하드웨어를 활용하여 성능을 극대화하는 설정입니다. <br> • **Use XNNPack (ON):** 부동 소수점(Floating-point) 모델을 위해 설계된 CPU 가속 라이브러리 <br> • **Use NNAPI Acceleration (OFF):** 안드로이드 신경망 API를 통해 하드웨어 가속(GPU, DSP, NPU 등)을 사용할지 여부
---