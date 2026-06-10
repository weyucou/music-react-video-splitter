"""Is-music heuristic bench (#17 follow-up).

Per-second features from the classic speech/music discrimination literature
(Scheirer & Slaney 1997): low-energy frame ratio + spectral flatness, combined
with Silero VAD speech delineation. Scored end-to-end through the identical
downstream pipeline as classifier_bench.py.

Usage:
    uv run python scripts/music_heuristic_bench.py [--tune VIDEO_ID]
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from classifier_bench import BENCH_DIR, PHASE0_CORPUS, fetch_metadata, parse_song_starts  # noqa: E402
from sanji.functions import (  # noqa: E402
    compute_music_density,
    find_song_regions,
    find_split_points,
    merge_short_regions,
)
from sanji.settings import DEFAULT_MIN_SONG, VALIDATION_TOLERANCE  # noqa: E402

SAMPLING_RATE = 16000
FRAME = 800  # 50 ms frames
FRAMES_PER_SECOND = SAMPLING_RATE // FRAME  # 20


def per_second_features(video_id: str) -> np.ndarray:
    """Return array [n_seconds, 3]: low_energy_ratio, spectral_flatness, rms."""
    cache = BENCH_DIR / f"{video_id}.features.npy"
    if cache.exists():
        return np.load(cache)

    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(BENCH_DIR / f"{video_id}.wav"), sampling_rate=SAMPLING_RATE)
    n_seconds = len(audio) // SAMPLING_RATE
    feats = np.zeros((n_seconds, 3), dtype=np.float32)

    for sec in range(n_seconds):
        chunk = audio[sec * SAMPLING_RATE : (sec + 1) * SAMPLING_RATE]
        frames = chunk[: FRAMES_PER_SECOND * FRAME].reshape(FRAMES_PER_SECOND, FRAME)
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        mean_rms = frame_rms.mean()
        low_energy_ratio = float(np.mean(frame_rms < 0.5 * mean_rms)) if mean_rms > 0 else 1.0

        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) + 1e-10
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))

        feats[sec] = (low_energy_ratio, flatness, float(mean_rms))

    np.save(cache, feats)
    return feats


def speech_seconds(video_id: str, n_seconds: int) -> np.ndarray:
    """Per-second VAD speech indicator (cached)."""
    cache = BENCH_DIR / f"{video_id}.speech.npy"
    if cache.exists():
        return np.load(cache)

    from sanji.audio_stt import classify_audio_stt

    segments = classify_audio_stt(BENCH_DIR / f"{video_id}.wav")
    is_speech = np.zeros(n_seconds, dtype=np.float32)
    for label, start, end in segments:
        if label == "speech":
            is_speech[int(start) : min(int(end) + 1, n_seconds)] = 1.0
    np.save(cache, is_speech)
    return is_speech


def classify_heuristic(
    video_id: str,
    low_energy_thresh: float = 0.25,
    rms_floor: float = 0.01,
) -> list[tuple[str, float, float]]:
    """Per-second labels: music when the energy distribution says music-dominant.

    A second is music when its low-energy frame ratio is below threshold (few
    quiet frames = continuous/mastered audio) AND it is loud enough to be the
    main program (rms above floor). Everything else: speech (if VAD) or noise.
    """
    feats = per_second_features(video_id)
    n_seconds = len(feats)
    is_speech = speech_seconds(video_id, n_seconds)

    low_energy, _flatness, rms = feats[:, 0], feats[:, 1], feats[:, 2]
    is_music = (low_energy < low_energy_thresh) & (rms > rms_floor)

    segments: list[tuple[str, float, float]] = []
    for sec in range(n_seconds):
        if is_music[sec]:
            label = "music"
        elif is_speech[sec]:
            label = "speech"
        else:
            label = "noise"
        segments.append((label, float(sec), float(sec + 1)))
    return segments


def score_boundaries(video_id: str, segments: list[tuple[str, float, float]]) -> dict:
    meta = fetch_metadata(video_id)
    total = meta["duration"]
    song_starts = [s for s, _ in parse_song_starts(meta["description"], total)]

    times, densities = compute_music_density(segments, total, window_size=60.0)
    regions = find_song_regions(times, densities, threshold=0.5, min_song_duration=30.0)
    regions = merge_short_regions(regions, min_duration=DEFAULT_MIN_SONG)
    split_points = find_split_points(regions, total)

    offsets = [min((abs(sp - t) for sp in split_points), default=float("inf")) for t in song_starts]
    matched = sum(1 for o in offsets if o <= VALIDATION_TOLERANCE)
    finite = sorted(o for o in offsets if o != float("inf"))
    return {
        "expected": len(song_starts),
        "matched": matched,
        "median_offset": round(finite[len(finite) // 2], 1) if finite else None,
        "regions": len(regions),
    }


def main() -> int:
    tune_id = None
    if "--tune" in sys.argv:
        tune_id = sys.argv[sys.argv.index("--tune") + 1]

    if tune_id:
        print(f"tuning low_energy_thresh on {tune_id}")
        for thresh in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            segments = classify_heuristic(tune_id, low_energy_thresh=thresh)
            s = score_boundaries(tune_id, segments)
            print(f"  thresh={thresh:.2f}  matched={s['matched']}/{s['expected']} median={s['median_offset']}s regions={s['regions']}")
        return 0

    report = {}
    for video_id in PHASE0_CORPUS:
        segments = classify_heuristic(video_id)
        s = score_boundaries(video_id, segments)
        report[video_id] = s
        print(f"{video_id}: matched={s['matched']}/{s['expected']} ({s['matched']/s['expected']:.0%}) median={s['median_offset']}s regions={s['regions']}")

    out = BENCH_DIR / "music_heuristic_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
