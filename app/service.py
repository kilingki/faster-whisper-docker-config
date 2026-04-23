from typing import Any

from faster_whisper import BatchedInferencePipeline, WhisperModel

from .config import (
    COMPUTE_TYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BEAM_SIZE,
    DEVICE,
    MODEL_ID,
)


class STTService:
    def __init__(self) -> None:
        self.model = WhisperModel(
            MODEL_ID,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        self.batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        beam_size: int | None = None,
    ) -> dict[str, Any]:
        beam = beam_size or DEFAULT_BEAM_SIZE

        segments_gen, info = self.batched_model.transcribe(
            audio_path,
            batch_size=DEFAULT_BATCH_SIZE,
            beam_size=beam,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = list(segments_gen)

        result: dict[str, Any] = {
            "text": "".join(seg.text for seg in segments).strip(),
            "language": getattr(info, "language", language),
            "duration": None,
            "segments": [],
        }

        for idx, seg in enumerate(segments):
            item: dict[str, Any] = {
                "id": idx,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": [],
            }

            if seg.words:
                for word in seg.words:
                    item["words"].append(
                        {
                            "start": float(word.start) if word.start is not None else None,
                            "end": float(word.end) if word.end is not None else None,
                            "word": word.word,
                            "probability": getattr(word, "probability", None),
                        }
                    )

            result["segments"].append(item)

        return result