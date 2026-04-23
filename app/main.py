import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .formatter import (
    to_json_response,
    to_srt,
    to_text_response,
    to_verbose_json_response,
    to_vtt,
)
from .service import STTService

app = FastAPI(
    title="OpenAI-Compatible STT Server",
    version="0.1.0",
)

stt_service = STTService()
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    beam_size: Optional[int] = Form(None),
):
    del prompt
    del temperature

    supported_formats = {"json", "text", "srt", "vtt", "verbose_json"}
    if response_format not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {response_format}",
        )

    accepted_models = {
        "whisper-1",
        "ghost613/faster-whisper-large-v3-turbo-korean",
        "whisper-large-v3-turbo-korean",
    }
    if model not in accepted_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}",
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name

        result = stt_service.transcribe(
            audio_path=tmp_path,
            language=language,
            beam_size=beam_size,
        )

        if response_format == "json":
            return JSONResponse(to_json_response(result))
        if response_format == "verbose_json":
            return JSONResponse(to_verbose_json_response(result))
        if response_format == "text":
            return PlainTextResponse(to_text_response(result), media_type="text/plain")
        if response_format == "srt":
            return PlainTextResponse(to_srt(result), media_type="text/plain")
        if response_format == "vtt":
            return PlainTextResponse(to_vtt(result), media_type="text/vtt")

        raise HTTPException(status_code=400, detail="Invalid response_format")
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass