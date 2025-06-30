import subprocess
import json

def export_openvino_model(
    model="microsoft/speecht5_tts",
    vocoder="microsoft/speecht5_hifigan",
    output_dir="speecht5_tts"
):
    print(f"📦 모델이 없어서 export 시작: {output_dir}")
    kwargs = json.dumps({"vocoder": vocoder})

    result = subprocess.run([
        "optimum-cli",
        "export", "openvino",
        "--model", model,
        "--model-kwargs", kwargs,
        output_dir
    ])

    if result.returncode != 0:
        raise RuntimeError("❌ 모델 export 실패")
