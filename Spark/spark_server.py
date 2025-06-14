import torch
import sys
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import soundfile as sf
from spark_inference import spark_inference
sys.path.append('catt')
from eo_pl import TashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = TashkeelTokenizer()
tashkeel_model = TashkeelModel(tokenizer, max_seq_len=1024, n_layers=6, learnable_pos_emb=False)
ckpt_path = 'catt/models/best_eo_mlm_ns_epoch_193.pt'
tashkeel_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
tashkeel_model.eval().to(DEVICE)

app = FastAPI()

# Enable CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/synthesize/spark")
async def synthesize(request: Request):
    data = await request.json()
    text = data["text"]
    
    spark_inference(text, tashkeel_model, tokenizer)

    audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output/output.wav'))
    if not os.path.exists(audio_path):
        return {"error": "Audio file not found"}
    audio, sr = sf.read(audio_path)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")