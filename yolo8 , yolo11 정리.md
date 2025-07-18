# YOLOv8 vs YOLOv11 비교 정리

## 📌 개요

| 항목 | YOLOv8 | YOLOv11 |
|------|--------|---------|
| 개발사 | Ultralytics | Ultralytics |
| 최초 릴리즈 | 2023년 1월 | 2024년 중반 |
| 프레임워크 | PyTorch 기반 | PyTorch 기반 |
| 라이센스 | GPL-3.0 | AGPL-3.0 |
| 사용 목적 | 실시간 객체 탐지, 분할, 추적 | 고정밀 탐지, 속도 향상, 경량화 |
| 지원 기능 | Detection, Segmentation, Classification, Pose | Detection, Segmentation, Multi-label Classification 등 확장 |

---

## 🔍 YOLOv8 개요

- **모듈형 구조**: 모델 아키텍처가 보다 깔끔하게 나뉘어져 있어 사용자 커스터마이징이 용이
- **벡터 기반 앵커 프리 구조**: YOLOv5~v7과 달리 앵커를 사용하지 않음
- **성능 향상**: COCO mAP 기준 기존 YOLOv5 대비 +2~3% 성능 증가
- **활용성**: CLI와 Python API 모두 제공

### 특징 요약
- Anchor-Free Detection
- Auto-labeling 지원
- TensorRT 및 ONNX export 가능
- Pose Estimation 지원

---

## 🔍 YOLOv11 개요

- **차세대 백본**: 자체 개발된 `V11Net` 백본 사용으로 연산 효율 향상
- **Neural Architecture Search (NAS)** 기반 구조 최적화
- **압축 및 경량화**: 모델 사이즈 대비 정확도 대폭 향상
- **다중 레이블 객체 탐지 및 분류** 기능 내장
- **BatchNorm2Fusion 등 추론 최적화 기법** 내장

### 특징 요약
- NAS 기반 구조
- Quantization-aware Training 지원
- Batch Normalization Fusion 내장
- Mobile/Edge 환경 최적화
- 다양한 post-processing 옵션 제공 (e.g. soft-NMS)

---

## ⚖️ 차이점 비교

| 비교 항목 | YOLOv8 | YOLOv11 |
|-----------|--------|---------|
| 백본 | CSPDarknet (변형) | V11Net (NAS 기반 설계) |
| 구조 최적화 | 수작업 튜닝 | NAS 자동 구조 탐색 |
| 앵커 방식 | Anchor-Free | Anchor-Free + Scale Adaptive |
| 추론 속도 | 빠름 | 더 빠름 (최적화 버전 기준) |
| 정확도 | 높음 | 더 높음 (COCO mAP 기준 +1~2%) |
| 경량화 모델
