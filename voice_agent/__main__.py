import array
import math
import os
import time
from pathlib import Path

from voice_kit import (
    ASR_FINAL, LLM_COMPLETE, LLM_FIRST_TOKEN,
    ReplaySession, read_trace, record_problems, summarize_stages,
)
from .asr import WhisperASR
from .llm import GPTOssLLM
from .tools import ToolBox

TTS_SAMPLE_RATE_HZ = 16000

def fake_tts(sentence):
    # 生成一段音量足够大的 440 Hz 正弦波，返回 16-bit PCM 字节
    sample_rate = TTS_SAMPLE_RATE_HZ
    duration = 0.5  # seconds
    frequency = 440  # Hz
    num_samples = int(sample_rate * duration)
    samples = array.array(
        "h",
        (
            int(8000 * math.sin(2 * math.pi * frequency * n / sample_rate))
            for n in range(num_samples)
        ),
    )
    if samples.itemsize != 2:
        raise RuntimeError("16-bit PCM requires 2-byte signed samples")
    return samples.tobytes()

def run_session(recordings_directory: Path, output_directory: Path, limit: int = 3) -> None:
    recordings = sorted(recordings_directory.glob("*.wav"))[:limit]   # 先只跑 3 个
    session = ReplaySession(
        recordings,
        output_directory,
        sink_sample_rate_hz=TTS_SAMPLE_RATE_HZ,   # 必须和 fake_tts 的采样率一致
    )
    asr = WhisperASR()
    asr.warm_up()   # 触发第一次推理的初始化开销，避免影响后续的延迟测量
    tools = ToolBox()   # 这一场对话的计时器等状态，LLM 通过工具调用来读写
    llm = GPTOssLLM.from_config(tools=tools)   # 设置在 voice_agent/config.yaml；环境变量 LLM_BASE_URL 等可覆盖
    llm.warm_up()   # 触发第一次推理的初始化开销，避免影响后续的延迟测量
    for turn in session.turns():
        received_audio = bytearray()
        turn.start(on_chunk=received_audio.extend)   # "按下按键"：shim 开始实时喂音频
        try:
            turn.end_of_speech_reached.wait()        # "松开按键"：shim 打 end_of_user_speech

            transcript = asr.transcribe(bytes(received_audio))
            turn.stamp(ASR_FINAL)
            turn.extras["transcript"] = transcript

            # llm.stream() 产出的是 token 片段，不是单词：片段自带前导空格（" green"），
            # 一个词也可能被切成几片（"St" + "eep"）。所以用 "" 直接拼接，不能用 " ".join。
            reply = ""      # 整个回复
            pending = ""    # 还没送去 TTS 的部分

            # llm.chat() 带上之前所有轮的对话，并在回答结束后把这一轮存进 llm.history
            for index, chunk in enumerate(llm.chat(transcript)):
                if index == 0:
                    turn.stamp(LLM_FIRST_TOKEN)
                reply += chunk
                pending += chunk
                # 看累积文本的结尾，而不是单个片段：句号可能单独成片，也可能后面跟着 "**" 或空格
                if pending.rstrip().rstrip("*\"')").endswith((".", "?", "!")):
                    turn.sink.write(fake_tts(pending.strip()))
                    pending = ""

            turn.stamp(LLM_COMPLETE)

            if pending.strip():     # 还有没送出去的半句
                turn.sink.write(fake_tts(pending.strip()))

            turn.extras["reply"] = reply
            turn.extras["tool_calls"] = llm.last_tool_calls

        finally:
            turn.finish()    # 不管成功还是出错，都写一条 trace 记录


def check(trace_path: Path) -> None:
    records = read_trace(trace_path)
    for record in records:
        for problem in record_problems(record):
            print(record["turn_id"], problem)
    print(summarize_stages(records))


if __name__ == "__main__":
    # One folder per run. sbatch passes its own in OUTPUT_DIR (it also holds the vLLM log); run by hand,
    # the agent names one by the start time, so the newest run sorts last.
    output_directory = Path(os.environ.get("OUTPUT_DIR")
                            or Path("run_output") / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run_session(Path("recordings"), output_directory, limit=None)   # None 表示跑所有录音
    check(output_directory / "trace.jsonl")
