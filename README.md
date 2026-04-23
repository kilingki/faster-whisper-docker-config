# faster-whisper-docker-config

This project provides a Docker Compose setup for running a Korean STT server based on `faster-whisper`. It exposes an OpenAI-style `/v1/audio/transcriptions` endpoint through FastAPI and is configured for GPU-based inference.

## Project Structure

- `docker-compose.yml`: runtime configuration for the STT server
- `Dockerfile`: CUDA-based container image definition
- `app/`: FastAPI app and Whisper service implementation
- `scripts/test_faster_whisper_youtube.py`: test script that downloads YouTube audio and sends it to the STT API

## Requirements

- NVIDIA GPU
- NVIDIA Container Toolkit
- Docker / Docker Compose

## Quick Start

1. Copy the example environment file.

```bash
cp .env.example .env
```

2. Review the host path used for the model cache in `.env`.

```env
MODEL_HOST_DIR=../models/stt
```

3. Build and start the container.

```bash
docker compose up --build -d
```

4. Check the health endpoint.

```bash
curl http://localhost:8001/health
```

Expected response:

```json
{"status":"ok"}
```

## API Example

```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "language=ko" \
  -F "response_format=verbose_json"
```

Supported `response_format` values:

- `json`
- `text`
- `srt`
- `vtt`
- `verbose_json`

## Environment Variables

The main settings are documented in `.env.example`.

- `STT_PORT`: host port exposed by Docker Compose
- `MODEL_HOST_DIR`: local directory used for model cache and downloads
- `MODEL_ID`: Hugging Face model ID
- `DEVICE`: for example, `cuda`
- `COMPUTE_TYPE`: for example, `float16`
- `DEFAULT_LANGUAGE`: default language code
- `DEFAULT_BEAM_SIZE`: default beam size
- `DEFAULT_BATCH_SIZE`: default batch size

## Test Script

The repository includes a test script that downloads audio from YouTube and transcribes the full audio in sequential chunks.

```bash
python3 scripts/test_faster_whisper_youtube.py
```

Set the YouTube URL directly at the top of `scripts/test_faster_whisper_youtube.py`, and adjust these values in `.env` if needed:

- `STT_BASE_URL`
- `STT_CHUNK_SECONDS`
- `STT_CHUNK_OVERLAP_SECONDS`
- `STT_OUTPUT_DIR`

Generated outputs are saved to `scripts/outputs/` by default, and that directory is excluded from git tracking.

## Notes

- The first run may take some time because the model needs to be downloaded.
- The current setup assumes GPU execution.
