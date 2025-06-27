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
    open("gradiod_helper.py", "w").write(r.text)

# Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
from notebook_utils import collect_telemetry

collect_telemetry("text-to-speech-genai.ipynb")

model_id = "microsoft/speecht5_tts"
model_path = Path(model_id.split("/")[-1])

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

################


device = "CPU"

import openvino_genai as ov_genai
import IPython.display as ipd


def play(data, rate=None):
    ipd.display(ipd.Audio(data, rate=rate))


pipe = ov_genai.Text2SpeechPipeline(model_path, device)


################################

import soundfile as sf

input_text = "It is not in the stars to hold our destiny but in ourselves."

result = pipe.generate(input_text)

speech = result.speeches[0]
output_file_name = "output_audio.wav"
sf.write(output_file_name, speech.data[0], samplerate=16000)

play(speech.data[0], rate=16000)

################################

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

###############################

from gradio_helper import make_demo

demo = make_demo(pipe)

try:
    demo.launch(share=True, debug=True, inbrowser=True)
except Exception:
    demo.launch(share=True, debug=True)

# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/