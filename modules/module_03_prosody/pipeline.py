from modules.module_03_prosody.extractor import ProsodyExtractor
from modules.module_03_prosody.baseline import ProsodyBaselineManager


# Create these once so baseline memory is not lost after every turn
prosody_extractor = ProsodyExtractor()
prosody_baseline_manager = ProsodyBaselineManager(baseline_turns=2)


def process_prosody_turn(
    audio_clip,
    turn_id,
    candidate_id,
    word_timestamps=None,
    response_latency_ms=None,
):
    """
    Full Module 3 Prosody pipeline.

    Step 1: Extract raw prosody features from audio.
    Step 2: Add candidate-specific baseline deviations.
    """

    raw_features = prosody_extractor.extract(
        audio_clip=audio_clip,
        word_timestamps=word_timestamps,
        response_latency_ms=response_latency_ms,
    )

    final_features = prosody_baseline_manager.update_with_baseline(
        candidate_id=candidate_id,
        turn_id=turn_id,
        prosody_features=raw_features,
    )

    return final_features