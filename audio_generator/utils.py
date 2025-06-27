import IPython.display as ipd

def play(data, rate=None):
    try:
        ipd.display(ipd.Audio(data, rate=rate))
    except Exception:
        pass  # 노트북이 아닐 때는 그냥 무시
