# voice_agent/asr.py
import numpy as np
from faster_whisper import WhisperModel

class WhisperASR:
    def __init__(self, model_size_or_path: str = "small.en", device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 4, beam_size: int = 1) -> None:
        self.model = WhisperModel(model_size_or_path, device=device,
                                  compute_type=compute_type, cpu_threads=cpu_threads)
        self.beam_size = beam_size

    def transcribe(self, pcm_bytes: bytes) -> str:
        audio = self._pcm16_to_float32(pcm_bytes)
        segments, _info = self.model.transcribe(audio, language="en", beam_size=self.beam_size)
        return " ".join(segment.text.strip() for segment in segments)   # ← 见下面的"关键点"

    def warm_up(self) -> None:
        self.transcribe(bytes(16000 * 2))   # 1 秒静音：触发第一次推理的初始化开销

    def _pcm16_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)     # 字节 → 整数，范围 -32768..32767
        return samples.astype(np.float32) / 32768.0            # 整数 → 浮点，范围 -1..1
