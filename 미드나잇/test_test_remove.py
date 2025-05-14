import whisper
import requests
import torch
import torchaudio
import os
import re
import sys
import io
import time

# ✅ 현재 경로 기준으로 ZonOS 모듈 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "Zonos-for-windows"))

# ✅ ZonOS 모듈 import
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# ✅ TorchDynamo 설정 (Inductor 오류 방지)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ✅ Whisper STT 수행
audio_path = os.path.join(current_dir, "my_korean_audio.m4a")
model_whisper = whisper.load_model("small")
result = model_whisper.transcribe(audio_path, language="ko")
stt_text = result["text"]
print("🎧 음성 인식 결과:", stt_text)

# ✅ Ollama LLM 호출
prompt = f"""당신은 바닷속을 탐험하며 만나는 지식 많은 인공지능 캐릭터입니다.
8~10세 어린이들이 궁금해하는 바닷속 생물, 지형, 모험에 대해 친절하고 따뜻하게 설명해줘요.

조건:  
- 이모지,이모티콘 사용 금지.
- 반복 문장 사용 금지.
- 쉬운 단어 사용
- 문장 짧게
- 친절하고 긍정적인 말투
- 때로는 의성어나 감탄사 활용

아이가 이렇게 말했어요:
\"{stt_text}\"

이 아이가 이해하기 쉽게, 즐겁게 대답해줘요!"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "gemma3:4b", "prompt": prompt, "stream": False}
)

# ✅ 예외 방지: 응답 키 체크
response_json = response.json()
print("📦 응답 전체 JSON:", response_json)

if "response" not in response_json:
    raise ValueError(f"❌ 'response' 키가 응답에 없습니다. 응답 내용: {response_json}")

ai_response = response_json["response"]
print("🤖 AI 응답:", ai_response)

# ✅ 텍스트 전처리 함수
def clean_for_tts(text: str) -> list:
    text = re.sub(r"[^\w\s,.?!가-힣]", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    lines = text.strip().splitlines()
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            cleaned.append(line)
            seen.add(line)
    return cleaned

cleaned_lines = clean_for_tts(ai_response)

# ✅ ZonOS 모델 로딩
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")

# ✅ 화자 임베딩 로딩 또는 생성
embedding_path = os.path.join(current_dir, "speaker_embedding_korean4.pt")
maru_path = os.path.join(current_dir, "Zonos-for-windows/assets/maru.wav")

if not os.path.exists(embedding_path):
    wav, sr = torchaudio.load(maru_path)
    speaker = model.make_speaker_embedding(wav, sr)
    torch.save(speaker, embedding_path)
    print("🎤 화자 임베딩 저장 완료:", embedding_path)
else:
    speaker = torch.load(embedding_path, map_location=device)
    print("📂 저장된 화자 임베딩 불러옴:", embedding_path)

# ✅ 문장 단위 TTS 생성 및 저장
output_dir = os.path.join(current_dir, "tts_chunks")
os.makedirs(output_dir, exist_ok=True)

for idx, line in enumerate(cleaned_lines):
    print(f"🎙️ [{idx+1}/{len(cleaned_lines)}] TTS 생성 중: {line}")
    start = time.time()

    cond_dict = make_cond_dict(
        text=line,
        speaker=speaker,
        language="ko",
        emotion=[0.8, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.3],
        speaking_rate=14.0,
        pitch_std=110
    )
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(conditioning)
    wav_tensor = model.autoencoder.decode(codes).cpu()[0]

    output_path = os.path.join(output_dir, f"tts_{idx+1:02d}.wav")
    torchaudio.save(output_path, wav_tensor, model.autoencoder.sampling_rate)

    print(f"✅ 저장 완료: {output_path} (⏱️ {time.time() - start:.2f}s)")

print("🎉 모든 문장 TTS 완료. 저장된 경로:", output_dir)
