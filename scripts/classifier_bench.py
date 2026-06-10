"""Benchmark: inaSpeechSegmenter vs Silero-VAD classifier (issue #17).

For each VOD, run BOTH classifiers through the identical downstream pipeline
(density -> regions -> merge -> split points) and compare detected boundaries
against song-start timestamps parsed from the video description.

Usage:
    uv run python scripts/classifier_bench.py [VIDEO_ID ...]

Defaults to the jawed Phase 0 corpus (@ryanmear LRQ VODs).
Audio and metadata are cached under data/bench/.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sanji.audio import classify_audio  # noqa: E402
from sanji.audio_stt import classify_audio_stt  # noqa: E402
from sanji.functions import (  # noqa: E402
    compute_music_density,
    find_song_regions,
    find_split_points,
    merge_short_regions,
)
from sanji.settings import (  # noqa: E402
    DEFAULT_MIN_SEGMENT,
    DEFAULT_MIN_SONG,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    VALIDATION_TOLERANCE,
)

PHASE0_CORPUS = ("Pbo9ScpNlOk", "2SbcBKkGVEQ", "dvhdL2emTSA")
BENCH_DIR = REPO_ROOT / "data" / "bench"

# Description format on the corpus is "M:SS Title" / "H:MM:SS Title" (no dash,
# which sanji.functions.parse_description_timestamps requires). Song entries
# carry curly-quoted titles; tangents/opening do not.
TIMESTAMP_LINE_RE = re.compile(r"^(\d+):(\d{2})(?::(\d{2}))?\s+(.+)$")


def fetch_metadata(video_id: str) -> dict:
    cache = BENCH_DIR / f"{video_id}.meta.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf8"))
    out = subprocess.run(
        ["uv", "run", "yt-dlp", "--skip-download", "--print", "%(duration)s\n===DESC===\n%(description)s",
         f"https://www.youtube.com/watch?v={video_id}"],
        check=True, capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout
    duration_raw, description = out.split("\n===DESC===\n", 1)
    meta = {"duration": float(duration_raw.strip()), "description": description}
    cache.write_text(json.dumps(meta), encoding="utf8")
    return meta


def fetch_audio(video_id: str) -> Path:
    wav = BENCH_DIR / f"{video_id}.wav"
    if wav.exists():
        return wav
    print(f"[{video_id}] downloading audio...")
    subprocess.run(
        ["uv", "run", "yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "wav",
         "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
         "-o", str(BENCH_DIR / f"{video_id}.%(ext)s"),
         f"https://www.youtube.com/watch?v={video_id}"],
        check=True, capture_output=True, cwd=REPO_ROOT,
    )
    assert wav.exists(), f"audio download failed for {video_id}"
    return wav


def parse_song_starts(description: str, total_duration: float) -> list[tuple[float, str]]:
    """Ground truth: timestamped lines whose title is a curly-quoted song entry."""
    songs = []
    for line in description.strip().split("\n"):
        match = TIMESTAMP_LINE_RE.match(line.strip())
        if not match:
            continue
        h_or_m, m_or_s, maybe_s, title = match.groups()
        if maybe_s is not None:
            seconds = int(h_or_m) * 3600 + int(m_or_s) * 60 + int(maybe_s)
        else:
            seconds = int(h_or_m) * 60 + int(m_or_s)
        if seconds > total_duration:
            continue
        if "“" in title:  # curly opening quote marks an actual song entry
            songs.append((float(seconds), title))
    return songs


def run_classifier(name: str, classify, audio_path: Path, total_duration: float) -> dict:
    started = time.monotonic()
    segments = classify(audio_path)
    classify_seconds = time.monotonic() - started

    times, densities = compute_music_density(segments, total_duration, window_size=DEFAULT_WINDOW_SIZE)
    regions = find_song_regions(times, densities, threshold=DEFAULT_THRESHOLD, min_song_duration=DEFAULT_MIN_SEGMENT)
    regions = merge_short_regions(regions, min_duration=DEFAULT_MIN_SONG)
    split_points = find_split_points(regions, total_duration)
    return {
        "name": name,
        "classify_seconds": round(classify_seconds, 1),
        "regions": len(regions),
        "split_points": split_points,
    }


def score(split_points: list[float], song_starts: list[float]) -> dict:
    """Boundary parity: a split should sit near each song start (excluding the first song,

    which has no preceding boundary when the VOD opens with it)."""
    expected = song_starts[1:] if song_starts and song_starts[0] < VALIDATION_TOLERANCE else song_starts
    if not expected:
        return {"expected": 0, "matched": 0, "median_offset": None}
    offsets = []
    matched = 0
    for target in expected:
        if not split_points:
            offsets.append(float("inf"))
            continue
        offset = min(abs(sp - target) for sp in split_points)
        offsets.append(offset)
        if offset <= VALIDATION_TOLERANCE:
            matched += 1
    finite = sorted(o for o in offsets if o != float("inf"))
    median = finite[len(finite) // 2] if finite else None
    return {
        "expected": len(expected),
        "matched": matched,
        "match_rate": round(matched / len(expected), 3),
        "median_offset": round(median, 1) if median is not None else None,
    }


def main() -> int:
    video_ids = sys.argv[1:] or list(PHASE0_CORPUS)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for video_id in video_ids:
        meta = fetch_metadata(video_id)
        audio_path = fetch_audio(video_id)
        song_starts_named = parse_song_starts(meta["description"], meta["duration"])
        song_starts = [s for s, _ in song_starts_named]
        print(f"\n[{video_id}] duration={meta['duration']:.0f}s songs={len(song_starts)}")

        entry: dict = {"duration": meta["duration"], "songs": len(song_starts)}
        for name, classify in (("silero_vad", classify_audio_stt), ("inaspeech", classify_audio)):
            result = run_classifier(name, classify, audio_path, meta["duration"])
            result["score"] = score(result["split_points"], song_starts)
            result["split_points"] = [round(sp, 1) for sp in result["split_points"]]
            entry[name] = result
            s = result["score"]
            print(
                f"  {name:12s} classify={result['classify_seconds']:7.1f}s regions={result['regions']:2d} "
                f"matched={s['matched']}/{s['expected']} ({s.get('match_rate', 0):.0%}) "
                f"median_offset={s['median_offset']}s"
            )
        report[video_id] = entry

    out = BENCH_DIR / "classifier_bench_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"\nReport written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
