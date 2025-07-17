# YOLOv8 vs YOLOv11 비교

## 1. 개요
- **YOLOv8 (Ultralytics, 2023)**  
  PyTorch 기반의 anchor-free 구조로, 객체 탐지, 분할, 포즈 추정 등 다양한 멀티태스크 지원.

- **YOLOv11 (Ultralytics, 2024년 9월 27일 발표)**  
  YOLO Vision 2024 행사에서 공식 발표됨 :contentReference[oaicite:1]{index=1}. 백본과 넥 구조가 개선되어 속도와 정확도 모두 진화.

---

## 2. 주요 특징 비교

| 항목                     | YOLOv8                            | YOLOv11                                 |
|------------------------|----------------------------------|-----------------------------------------|
| **출시 시점**            | 2023년                           | 2024년 9월 27일 :contentReference[oaicite:2]{index=2} |
| **네트워크 구조**       | CSP 백본, anchor-free head       | C3k2, SPPF, PSA 도입된 향상된 백본/넥 :contentReference[oaicite:3]{index=3} |
| **성능 (COCO mAP)**     | v8m 기준 약 51–54%              | v11m 기준 51.5%, v11x 기준 54.7% :contentReference[oaicite:4]{index=4} |
| **속도 및 효율**        | 상당히 빠름                      | v11m은 v8m 대비 파라미터 22% 감소하면서 속도↑ :contentReference[oaicite:5]{index=5} |
| **지원 모델 스케일**     | nano/s/m/l/x                     | n/s/m/l/x (탐지, 분할, 포즈, OBB 등) :contentReference[oaicite:6]{index=6} |
| **지원 태스크**         | Detection, Segmentation, Pose   | 단일탐지 외에 분할, 포즈, OBB, 분류, 추적까지 :contentReference[oaicite:7]{index=7} |
| **추론 환경**           | ONNX, TensorRT, CoreML 등 지원  | 엣지·클라우드·GPU에 최적화, TensorRT, ONNX 등 지원 :contentReference[oaicite:8]{index=8} |
| **모델 크기 및 FLOPs**  | --                               | v11m: 20M params / 68 GFLOPs 등 :contentReference[oaicite:9]{index=9} |
| **라이선스**            | AGPL-3.0                          | AGPL‑3.0 및 엔터프라이즈 옵션 :contentReference[oaicite:10]{index=10} |

---

## 3. 장단점 요약

- **🎯 정확도 & 효율성**  
  YOLOv11은 파라미터를 줄이면서도 동일하거나 더 높은 mAP 성능 달성 :contentReference[oaicite:11]{index=11}.

- **⚡ 속도**  
  GPU 최적화로 추론 지연시간 감소 (v11n은 COCO 기반 56 FPS 이상, 파이썬 벤치 등) :contentReference[oaicite:12]{index=12}.

- **🧩 더 확장된 태스크 지원**  
  탐지, 분할, 포즈, OBB, 분류, 추적 등 다중 작업 프레임워크 제공 :contentReference[oaicite:13]{index=13}.

- **📦 배포 호환성**  
  ONNX나 TensorRT 외에도 엣지 장치 지원 및 클라우드 최적화 강화 :contentReference[oaicite:14]{index=14}.

---

## 4. 실제 사용자 반응 & 벤치마크

- Reddit에서는 성능 향상이 있지만 “2% 정도”로, 효율성(연산량 감소)에 더 큰 의미 부여 :contentReference[oaicite:15]{index=15}:

  > “It's only about 2% better, but requires less compute to do it.”

- 전력 장비 탐지 테스트에서는 YOLOv11이 v8 대비 최고 성능을 기록 (mAP 57.2%) :contentReference[oaicite:16]{index=16}.

---

## 5. 결론

- **YOLOv8**: 이미 충분한 정확도와 속도를 제공하며, 사용성과 안정성이 뛰어남.
- **YOLOv11**: 구조적 개선과 효율 최적화로 **더 강력하고 다목적**, 특히 리소스 제약 환경이나 복합 태스크에 적합.

---

## 6. 참고 및 인용

- Ultralytics 공식 문서 및 출시 정보 :contentReference[oaicite:17]{index=17}  
- 구조 설명 (C3k2, SPPF, PSA) :contentReference[oaicite:18]{index=18}  
- 벤치마크 및 라이센스 정보 :contentReference[oaicite:19]{index=19}
