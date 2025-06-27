# OpenVINO 텍스트 음성 변환

이 프로젝트는 `microsoft/speecht5_tts` 모델을 활용하여 OpenVINO 툴킷을 사용한 텍스트 음성 변환(TTS)을 시연합니다. 또한 음성 사용자 정의를 위한 화자 임베딩 적용 방법과 대화형 데모를 위한 Gradio 인터페이스를 보여줍니다.

## 목차
- [OpenVINO 텍스트 음성 변환](#openvino-텍스트-음성-변환)
  - [목차](#목차)
  - [개요](#개요)
  - [설정](#설정)
  - [모델 내보내기](#모델-내보내기)
  - [사용법](#사용법)
    - [기본 텍스트 음성 변환](#기본-텍스트-음성-변환)
    - [화자 임베딩을 사용한 텍스트 음성 변환](#화자-임베딩을-사용한-텍스트-음성-변환)
    - [Gradio를 이용한 대화형 데모](#gradio를-이용한-대화형-데모)
  - [원격 측정 데이터 수집](#원격-측정-데이터-수집)

## 개요

이 프로젝트는 `openvino_genai` 라이브러리를 사용하여 텍스트 음성 합성을 수행합니다. 필요한 유틸리티 스크립트를 다운로드하고 `microsoft/speecht5_tts` 모델을 OpenVINO IR 형식으로 내보낸 다음 텍스트에서 음성을 생성하는 데 사용법을 보여주며, 보다 개인화된 오디오 출력을 위해 화자 임베딩을 포함하는 옵션을 제공합니다. 또한 손쉬운 실험을 위해 대화형 Gradio 데모도 제공됩니다.

## 설정

노트북을 실행하기 전에 필요한 파일이 있는지 확인하십시오. [cite_start]스크립트는 `notebook_utils.py` 및 `gradio_helper.py`가 없으면 자동으로 다운로드합니다.

```python
from pathlib import Path
import requests

if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="[https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py](https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py)",
    )
    open("notebook_utils.py", "w").write(r.text)


if not Path("gradio_helper.py").exists():
    r = requests.get(
        url="[https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/text-to-speech-genai/gradio_helper.py](https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/text-to-speech-genai/gradio_helper.py)",
    )
    open("gradiod_helper.py", "w").write(r.text)
