"""
BizForge AI — FastAPI Backend
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── LOAD ENV FIRST (before any other imports that read os.environ) ────────────
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from ai_services import (
    generate_brand_names,
    generate_logo_prompt,
    generate_marketing_content,
    analyze_sentiment,
    get_color_palette,
    chat_with_ai,
    generate_banner,
    FRONTEND,
    LOGOS_DIR,
)

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BizForge AI", version="1.0.0")

# ── CORS (FIX: was missing — caused browser fetch errors) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── STATIC FILES (FIX: path aligned with LOGOS_DIR in ai_services) ───────────
static_dir = FRONTEND / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT / BRANDING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/generate-brand")
async def generate_brand_endpoint(request: dict):
    try:
        result = await generate_brand_names(
            industry=request.get("industry", "General"),
            keywords=request.get("keywords", ""),
            tone=request.get("tone", "Professional"),
            language=request.get("language", "en"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-logo")
async def generate_logo_endpoint(request: dict):
    try:
        result = await generate_logo_prompt(
            brand_name=request.get("brand_name", "Brand"),
            industry=request.get("industry", "General"),
            keywords=request.get("keywords", "modern, professional"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-content")
async def generate_content_endpoint(request: dict):
    try:
        result = await generate_marketing_content(
            brand_description=request.get("brand_description", ""),
            tone=request.get("tone", "Professional"),
            content_type=request.get("content_type", "product_description"),
            language=request.get("language", "en"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-sentiment")
async def analyze_sentiment_endpoint(request: dict):
    try:
        result = await analyze_sentiment(
            text=request.get("text", ""),
            brand_tone=request.get("brand_tone", "Professional"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/get-colors")
async def get_colors_endpoint(request: dict):
    try:
        result = await get_color_palette(
            tone=request.get("tone", "Professional"),
            industry=request.get("industry", "Technology"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: dict):
    try:
        result = await chat_with_ai(request.get("message", "Hello"))
        return {"success": True, "data": {"content": result}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/banner")
async def generate_banner_endpoint(request: dict):
    try:
        result = await generate_banner(
            platform=request.get("platform", "instagram"),
            brand_name=request.get("brand_name", "Brand"),
            tagline=request.get("tagline", ""),
            cta=request.get("cta", ""),
            style=request.get("style", "modern"),
            output_format=request.get("output_format", "png"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/transcribe-voice")
async def transcribe_voice(audio_file: UploadFile = File(...)):
    """Transcribe audio using Google Speech-to-Text (free, no API key needed)."""
    try:
        import speech_recognition as sr
        from io import BytesIO

        audio_content = await audio_file.read()
        tmp_path = "/tmp/temp_audio.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_content)

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return {"success": True, "text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "groq":    bool(os.getenv("GROQ_API_KEY")),
        "hf":      bool(os.getenv("HF_API_KEY")),
        "version": "1.0.0",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FRONTEND SERVING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def serve_index():
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "BizForge API running. Frontend not found at " + str(FRONTEND)})


@app.get("/{page}.html")
async def serve_page(page: str):
    path = FRONTEND / f"{page}.html"
    if path.exists():
        return FileResponse(str(path))
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/{path:path}")
async def catch_all(path: str):
    file_path = FRONTEND / path
    if file_path.exists():
        return FileResponse(str(file_path))
    return FileResponse(str(FRONTEND / "index.html"))


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print("\n" + "=" * 60)
    print("🚀 BizForge AI Backend Started!")
    print("=" * 60)
    print(f"🌐 API running at  : http://localhost:8000")
    print(f"📁 Frontend path   : {FRONTEND}")
    print(f"🖼  Logos saved to  : {LOGOS_DIR}")
    print(f"🔑 Groq key set    : {'✅' if os.getenv('GROQ_API_KEY') else '❌ NOT SET'}")
    print(f"🔑 HF key set      : {'✅' if os.getenv('HF_API_KEY')   else '❌ NOT SET'}")
    print("=" * 60 + "\n")

@app.post("/api/generate-ad-creative")
async def ad_creative_endpoint(request: dict):
    try:
        from ai_services import generate_ad_creative
        result = await generate_ad_creative(
            brand_name=request.get("brand_name", "Brand"),
            product=request.get("product", ""),
            target_audience=request.get("target_audience", "General"),
            platform=request.get("platform", "Instagram"),
            goal=request.get("goal", "Awareness"),
            tone=request.get("tone", "Professional"),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/branding-assistant")
async def branding_assistant_endpoint(request: dict):
    try:
        from ai_services import branding_assistant
        result = await branding_assistant(
            brand_name=request.get("brand_name", "Brand"),
            industry=request.get("industry", "General"),
            stage=request.get("stage", "Startup"),
            request_type=request.get("request_type", "full_audit"),
            custom_question=request.get("custom_question", ""),
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
