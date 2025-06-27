import subprocess
import json

def export_openvino_model(model="microsoft/speecht5_tts", vocoder="microsoft/speecht5_hifigan", output_dir="speecht5_tts"):
    kwargs = json.dumps({"vocoder": vocoder})
    subprocess.run([
        "optimum-cli",
        "export", "openvino",
        "--model", model,
        "--model-kwargs", kwargs,
        output_dir
    ])