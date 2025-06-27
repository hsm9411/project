# OpenVINO 텍스트 음성 변환

이 프로젝트는 `microsoft/speecht5_tts` 모델을 활용하여 OpenVINO 툴킷을 사용한 텍스트 음성 변환(TTS)을 시연합니다. 또한 음성 사용자 정의를 위한 화자 임베딩 적용 방법과 대화형 데모를 위한 Gradio 인터페이스를 제공합니다.

## 목차
- [개요](#개요)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
  - [기본 실행](#기본-실행)
  - [모델 내보내기](#모델-내보내기)
  - [텍스트 음성 변환](#텍스트-음성-변환)
  - [Gradio 데모 실행](#gradio-데모-실행)
- [모듈 설명](#모듈-설명)

## 개요

이 프로젝트는 `openvino_genai` 라이브러리를 사용하여 텍스트 음성 합성을 수행합니다. 패키지화된 구조로 설계되어 있으며, 모델 내보내기부터 음성 생성, 대화형 데모까지 전체 TTS 파이프라인을 제공합니다.

## 프로젝트 구조

```
project/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 의존성 패키지 목록
├── download_basic.py       # 기본 파일 다운로드 스크립트
└── audio_generator/        # 메인 패키지
    ├── __init__.py
    ├── core.py            # 핵심 TTS 기능
    ├── gradio_runner.py   # Gradio 인터페이스 실행
    ├── openvino_export.py # 모델 내보내기 기능
    └── utils.py           # 유틸리티 함수
```

## 설치 및 설정

### 1. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 필수 파일 다운로드

프로젝트 실행에 필요한 `gradio_helper.py`와 `notebook_utils.py` 파일을 다운로드합니다.

```bash
python download_basic.py
```

## 사용법

### 기본 실행

메인 파일을 실행하여 전체 TTS 파이프라인을 시작합니다.

```bash
python main.py
```

### 모델 내보내기

`microsoft/speecht5_tts` 모델을 OpenVINO IR 형식으로 내보냅니다.

```python
from audio_generator.openvino_export import export_model

# 모델 내보내기
export_model()
```

내부적으로 다음과 같은 작업을 수행합니다:

```python
import subprocess
import json

kwargs = json.dumps({"vocoder": "microsoft/speecht5_hifigan"})

subprocess.run([
    "optimum-cli",
    "export", "openvino",
    "--model", "microsoft/speecht5_tts",
    "--model-kwargs", kwargs,
    "speecht5_tts"
])
```

### 텍스트 음성 변환

#### 기본 음성 생성

```python
from audio_generator.core import TextToSpeechGenerator

# TTS 생성기 초기화
tts_generator = TextToSpeechGenerator()

# 기본 음성 생성
input_text = "It is not in the stars to hold our destiny but in ourselves."
output_file = tts_generator.generate_speech(input_text)
```

#### 화자 임베딩을 사용한 음성 생성

```python
# 화자 임베딩과 함께 음성 생성
output_file = tts_generator.generate_speech_with_speaker(
    input_text, 
    speaker_id=7306
)
```

### Gradio 데모 실행

대화형 웹 인터페이스를 통해 TTS 기능을 테스트할 수 있습니다.

```python
from audio_generator.gradio_runner import run_demo

# Gradio 데모 실행
run_demo()
```

또는 직접 실행:

```bash
python -m audio_generator.gradio_runner
```

## 모듈 설명

### `audio_generator/core.py`
- 핵심 TTS 기능을 담당하는 메인 클래스
- OpenVINO 파이프라인 초기화 및 관리
- 기본 음성 생성 및 화자 임베딩 기반 음성 생성

### `audio_generator/openvino_export.py`
- HuggingFace 모델을 OpenVINO IR 형식으로 내보내기
- 모델 최적화 및 변환 관리

### `audio_generator/gradio_runner.py`
- Gradio 기반 웹 인터페이스 실행
- 사용자 친화적인 TTS 데모 제공

### `audio_generator/utils.py`
- 공통 유틸리티 함수들
- 파일 I/O, 오디오 처리 등의 보조 기능

### `download_basic.py`
- 필수 외부 파일 다운로드 관리
- `gradio_helper.py`, `notebook_utils.py` 자동 다운로드

### `main.py`
- 전체 애플리케이션의 진입점
- 각 모듈들을 조합하여 완전한 TTS 파이프라인 실행

## 요구사항

주요 의존성 패키지:
- `openvino-genai`
- `soundfile`
- `datasets`
- `numpy`
- `optimum[openvino]`
- `gradio`
- `requests`

전체 요구사항은 `requirements.txt`를 참조하세요.

## 참고사항

- 모델은 CPU에서 실행되도록 기본 설정되어 있습니다
- 화자 임베딩은 "Matthijs/cmu-arctic-xvectors" 데이터셋을 사용합니다
- 생성된 오디오는 16kHz 샘플링 레이트로 저장됩니다
