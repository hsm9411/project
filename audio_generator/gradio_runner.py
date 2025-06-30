# audio_generator/gradio_runner.py

from .core import AudioGenerator
from .utils import restore_spacing
import gradio as gr

generator = AudioGenerator()

def inference(user_input):
    restored = restore_spacing(user_input)
    print(f"🧠 복원된 문장: {restored}")
    audio_path = generator.generate(restored)
    return audio_path, restored  # 복원된 문장도 함께 출력

def launch_demo():
    demo = gr.Interface(
        fn=inference,
        inputs=gr.Textbox(lines=2, label="📥 영어 입력"),
        outputs=[
            gr.Audio(label="📤 출력 음성"),
            gr.Textbox(label="📝 복원된 문장 (TTS 입력)")
        ],
        title="🗣️ 영어 TTS + 띄어쓰기 복원"
    )
    demo.launch(inbrowser=True, share=True)
