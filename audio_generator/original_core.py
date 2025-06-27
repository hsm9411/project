import soundfile as sf
import openvino_genai as ov_genai
import numpy as np
import openvino as ov
from datasets import load_dataset

from .utils import play

class AudioGenerator:
    def __init__(self, model_dir="speecht5_tts", device="CPU"):
        self.pipe = ov_genai.Text2SpeechPipeline(model_dir, device)
        self.sr = 16000

        # 스피커 임베딩 불러오기 (고정)
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.embedding = ov.Tensor(np.array(dataset[7306]["xvector"], dtype=np.float32).reshape(1, -1))

    def generate(self, text, output="output_audio.wav", use_embedding=True):
        if use_embedding:
            result = self.pipe.generate(text, self.embedding)
        else:
            result = self.pipe.generate(text)

        speech = result.speeches[0]
        sf.write(output, speech.data[0], samplerate=self.sr)
        play(speech.data[0], rate=self.sr)