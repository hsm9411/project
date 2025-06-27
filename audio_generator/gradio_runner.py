from gradio_helper import make_demo
from .core import AudioGenerator

def launch_demo(share=True, inbrowser=True, debug=True, server_name=None, server_port=None):
    generator = AudioGenerator()
    pipe = generator.pipe

    demo = make_demo(pipe)

    launch_kwargs = dict(share=share, inbrowser=inbrowser, debug=debug)

    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = server_port

    try:
        demo.launch(**launch_kwargs)
    except Exception:
        demo.launch(share=True, debug=True)
