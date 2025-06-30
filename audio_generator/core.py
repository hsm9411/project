from pathlib import Path
import soundfile as sf
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from datasets import load_dataset

from .utils import play
from .openvino_export import export_openvino_model  # ✅ 이 라인만 바뀜


class AudioGenerator:
    def __init__(self, model_dir="speecht5_tts", device="CPU"):
        model_path = Path(model_dir)
        if not (model_path / "config.json").exists():
            export_openvino_model(output_dir=model_dir)

        self.pipe = ov_genai.Text2SpeechPipeline(str(model_path.resolve()), device)
        self.sr = 16000

        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.embedding = ov.Tensor(
            np.array(dataset[7306]["xvector"], dtype=np.float32).reshape(1, -1)
        )

    def generate(self, text, output="output_audio.wav", use_embedding=True):
        if use_embedding:
            result = self.pipe.generate(text, self.embedding)
        else:
            result = self.pipe.generate(text)

        speech = result.speeches[0]
        sf.write(output, speech.data[0], samplerate=self.sr)
        play(speech.data[0], rate=self.sr)
        print(f"✅ 생성 완료: {output}")
        return output
