import IPython.display as ipd
from wordsegment import load, segment

def play(data, rate=None):
    try:
        ipd.display(ipd.Audio(data, rate=rate))
    except Exception:
        pass  # 노트북이 아닐 때는 그냥 무시

load()

def restore_spacing(text: str) -> str:
    words = segment(text)
    return " ".join(words)