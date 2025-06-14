from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import soundfile as sf
from fastpitch_inference import fastpitch_infer_plain_arabic

app = FastAPI()

# Enable CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/synthesize")
async def synthesize(request: Request):
    data = await request.json()
    text = data.get("text")
    model_type = data.get("model_type", "fastpitch").lower()  # default to fastpitch 

    try:
        audio = fastpitch_infer_plain_arabic(text, model_type)
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    buf = io.BytesIO()
    sf.write(buf, audio, 22050, format='WAV')
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")