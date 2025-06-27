# OpenVINO 텍스트 음성 변환

이 프로젝트는 `microsoft/speecht5_tts` 모델을 활용하여 OpenVINO 툴킷을 사용한 텍스트 음성 변환(TTS)을 시연합니다. 또한 음성 사용자 정의를 위한 화자 임베딩 적용 방법과 대화형 데모를 위한 Gradio 인터페이스를 보여줍니다.

## 목차
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

### 필수 파일 다운로드

노트북을 실행하기 전에 필요한 파일이 있는지 확인하십시오. 스크립트는 `notebook_utils.py` 및 `gradio_helper.py`가 없으면 자동으로 다운로드합니다.

```python
from pathlib import Path
import requests

if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

if not Path("gradio_helper.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/text-to-speech-genai/gradio_helper.py",
    )
    open("gradio_helper.py", "w").write(r.text)
```

### 패키지 설치

`openvino-genai`, `soundfile`, `datasets`, `numpy`가 설치되어 있어야 합니다. 이러한 라이브러리는 일반적으로 pip를 통해 설치할 수 있습니다:

```bash
pip install openvino-genai soundfile datasets numpy optimum[openvino]
```

## 모델 내보내기

`microsoft/speecht5_tts` 모델은 `optimum-cli`를 사용하여 OpenVINO IR (Intermediate Representation) 형식으로 내보내집니다. 이 과정에는 `vocoder`를 `microsoft/speecht5_hifigan`으로 지정하는 것이 포함됩니다.

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

내보낸 모델은 `speecht5_tts`라는 디렉토리에 저장됩니다.

## 사용법

이 프로젝트는 텍스트 음성 변환 파이프라인을 사용하는 세 가지 주요 방법을 보여줍니다: 기본 생성, 화자 임베딩을 사용한 생성 및 대화형 Gradio 데모.

### 파이프라인 초기화

모델 실행을 위해 `device`는 "CPU"로 설정됩니다.

```python
device = "CPU"

import openvino_genai as ov_genai
import IPython.display as ipd

def play(data, rate=None):
    ipd.display(ipd.Audio(data, rate=rate))

pipe = ov_genai.Text2SpeechPipeline("speecht5_tts", device)
```

### 기본 텍스트 음성 변환

화자별 사용자 지정 없이 주어진 입력 텍스트에서 음성을 생성합니다. 출력 오디오는 `output_audio.wav`로 저장됩니다.

```python
import soundfile as sf

input_text = "It is not in the stars to hold our destiny but in ourselves."
result = pipe.generate(input_text)

speech = result.speeches[0]
output_file_name = "output_audio.wav"
sf.write(output_file_name, speech.data[0], samplerate=16000)

play(speech.data[0], rate=16000)
```

### 화자 임베딩을 사용한 텍스트 음성 변환

특정 음성으로 음성을 합성하기 위해 화자 임베딩을 사용하여 음성을 생성합니다. 화자 임베딩은 "Matthijs/cmu-arctic-xvectors" 데이터 세트에서 로드됩니다. 출력 오디오는 `output_audio_with_speaker.wav`로 저장됩니다.

```python
from datasets import load_dataset
import numpy as np
import openvino as ov

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = ov.Tensor(np.array(embeddings_dataset[7306]["xvector"], dtype=np.float32).reshape(1, -1))

result = pipe.generate(input_text, speaker_embedding)

speech = result.speeches[0]
output_file_name = "output_audio_with_speaker.wav"
sf.write(output_file_name, speech.data[0], samplerate=16000)

play(speech.data[0], rate=16000)
```

### Gradio를 이용한 대화형 데모

텍스트 음성 변환 기능을 대화형으로 시연하기 위한 Gradio 기반 웹 인터페이스가 제공됩니다.

```python
from gradio_helper import make_demo

demo = make_demo(pipe)

try:
    demo.launch(share=True, debug=True, inbrowser=True)
except Exception:
    demo.launch(share=True, debug=True)

# 원격으로 실행하려면 다음 줄의 주석을 제거하고 수정하십시오:
# demo.launch(server_name='your server name', server_port='server port in int')
# 자세한 내용은 문서(https://gradio.app/docs/)를 참조하십시오.
```

## 원격 측정 데이터 수집

이 프로젝트에는 사용 통계를 수집하기 위해 `notebook_utils.py`를 통한 원격 측정 데이터 수집이 포함됩니다. 원격 측정 및 옵트아웃 방법에 대한 자세한 내용은 OpenVINO Notebooks 저장소를 참조하십시오.

```python
from notebook_utils import collect_telemetry

collect_telemetry("text-to-speech-genai.ipynb")
```
