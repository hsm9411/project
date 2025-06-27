# from audio_generator import AudioGenerator

# generator = AudioGenerator()

# generator.generate("Who are you? who am I? and you? nice to meet you. my name is Pat Gelsinger")

from audio_generator.gradio_runner import launch_demo

if __name__ == "__main__":
    launch_demo()
