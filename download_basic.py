from pathlib import Path
import requests

# if not Path("notebook_utils.py").exists():
#     r = requests.get(
#         url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
#     )
#     open("notebook_utils.py", "w").write(r.text)


if not Path("gradio_helper.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/text-to-speech-genai/gradio_helper.py",
    )
    open("gradiod_helper.py", "w").write(r.text)

