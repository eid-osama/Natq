from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import soundfile as sf
from fastspeech2_inference import fastspeech2_infer

app = FastAPI()

# Enable CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/synthesize/fastspeech2")
async def synthesize(request: Request):
    data = await request.json()
    text = data["text"]
    fastspeech2_infer(text)
    audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'FastSpeech2/sample.wav'))
    if not os.path.exists(audio_path):
        return {"error": "Audio file not found"}
    audio, sr = sf.read(audio_path)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")