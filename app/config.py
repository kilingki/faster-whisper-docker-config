import os

MODEL_ID = os.getenv("MODEL_ID", "ghost613/faster-whisper-large-v3-turbo-korean")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ko")
DEFAULT_BEAM_SIZE = int(os.getenv("DEFAULT_BEAM_SIZE", "5"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "2"))

#MODEL_DIR = os.getenv("MODEL_DIR", "/models")