"""STT/VAD-based audio classification (issue #17 prototype).

Drop-in alternative to ``sanji.audio.classify_audio`` built on the Silero VAD
bundled with faster-whisper. Returns the same ``[(label, start, end), ...]``
contract: speech segments labeled "speech"; the gaps between them labeled
"music" (or "noEnergy" when near-silent), which is what
``compute_music_density`` consumes via MUSIC_LABELS.

No TensorFlow: interval detection happens in post-processing over the VAD
timeline, viable because splitting is batch VOD work, not real-time.
"""

from pathlib import Path

import numpy as np

SAMPLING_RATE = 16000
NO_ENERGY_RMS_THRESHOLD = 0.0015


def classify_audio_stt(audio_path: Path) -> list[tuple[str, float, float]]:
    """Run Silero VAD (via faster-whisper) and return [(label, start, end), ...]."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    print("Classifying audio with Silero VAD (post-processing intervals)...")
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLING_RATE)
    total_seconds = len(audio) / SAMPLING_RATE

    speech_chunks = get_speech_timestamps(audio, VadOptions())

    segments: list[tuple[str, float, float]] = []
    cursor = 0.0
    for chunk in speech_chunks:
        start = chunk["start"] / SAMPLING_RATE
        end = chunk["end"] / SAMPLING_RATE
        if start > cursor:
            segments.append((_gap_label(audio, cursor, start), cursor, start))
        segments.append(("speech", start, end))
        cursor = end
    if cursor < total_seconds:
        segments.append((_gap_label(audio, cursor, total_seconds), cursor, total_seconds))

    print(f"Classification complete: {len(segments)} segments")
    return segments


def _gap_label(audio: np.ndarray, start_seconds: float, end_seconds: float) -> str:
    """Label a non-speech gap as music or near-silence based on RMS energy."""
    chunk = audio[int(start_seconds * SAMPLING_RATE) : int(end_seconds * SAMPLING_RATE)]
    if chunk.size == 0:
        return "noEnergy"
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return "music" if rms >= NO_ENERGY_RMS_THRESHOLD else "noEnergy"
