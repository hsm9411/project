import openvino_genai as ov_genai
from pathlib import Path
import sys

from llm_config import convert_and_compress_model


class AudioGenerator:
    
    def __init__(self):
        model_id = "microsoft/speecht5_tts"
        model_path = Path(model_id.split("/")[-1])
        device = "CPU"

        model_dir = convert_and_compress_model(
            model_id, model_configuration, compression_variant, use_preconverted=True
        )



        self.pipe = ov_genai.LLMPipeline(str(model_dir), device)
        if "genai_chat_template" in model_configuration:
            self.pipe.get_tokenizer().set_chat _template(model_configuration["genai_chat_template"])

    
    def generate(self, prompt: str) -> str:
        def streamer(subword):
            self.mode
            print(subword, end="", flush=True)
            sys.stdout.flush()
            # Return flag corresponds whether generation should be stopped.
            # False means continue generation.
            return False

        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 128

        result = self.pipe.generate(prompt, generation_config, streamer)
        return result
    
    # ---end---

