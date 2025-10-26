"""Simple IoU-based tracker that keeps the best plate per vehicle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from yolo_ocr.config import TrackerConfig
from yolo_ocr.pipeline import postprocess
from yolo_ocr.pipeline.postprocess import PlatePrediction


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    best_confidence: float
    best_text: str
    last_confidence: float
    last_text: str
    lost_frames: int = 0


class PlateTracker:
    """Associates detections across frames and preserves highest-confidence OCR."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: Dict[int, _TrackState] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Remove all active tracks."""

        self._tracks.clear()
        self._next_id = 1

    def _match_track(self, prediction: PlatePrediction) -> Optional[_TrackState]:
        """Return the existing track that best matches ``prediction`` by IoU."""

        best_track: Optional[_TrackState] = None
        best_iou = 0.0
        for track in self._tracks.values():
            overlap = postprocess.iou(prediction.bbox, track.bbox)
            if overlap < self.config.match_iou_threshold:
                continue
            if overlap > best_iou:
                best_iou = overlap
                best_track = track
        return best_track

    def _create_track(self, prediction: PlatePrediction) -> _TrackState:
        track = _TrackState(
            track_id=self._next_id,
            bbox=prediction.bbox,
            class_name=prediction.class_name,
            best_confidence=prediction.confidence if prediction.text else 0.0,
            best_text=prediction.text if prediction.text else "",
            last_confidence=prediction.confidence,
            last_text=prediction.text,
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
        return track

    def update(self, predictions: List[PlatePrediction]) -> List[PlatePrediction]:
        """Update active tracks and return tracked predictions for this frame."""

        outputs: List[PlatePrediction] = []
        matched_ids: set[int] = set()

        for prediction in predictions:
            if prediction.confidence < self.config.min_init_confidence:
                # Skip extremely low-confidence detections entirely.
                continue

            track = self._match_track(prediction)
            if track is None:
                track = self._create_track(prediction)

            track.bbox = prediction.bbox
            track.last_confidence = prediction.confidence
            track.last_text = prediction.text
            track.lost_frames = 0

            if prediction.text and prediction.confidence >= track.best_confidence:
                track.best_confidence = prediction.confidence
                track.best_text = prediction.text

            outputs.append(
                PlatePrediction(
                    bbox=track.bbox,
                    confidence=track.best_confidence if track.best_text else track.last_confidence,
                    class_name=track.class_name,
                    text=track.best_text if track.best_text else track.last_text,
                    track_id=track.track_id,
                )
            )
            matched_ids.add(track.track_id)

        expired: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_ids:
                continue
            track.lost_frames += 1
            if track.lost_frames > self.config.max_lost_frames:
                expired.append(track_id)
                continue
            if track.best_text:
                outputs.append(
                    PlatePrediction(
                        bbox=track.bbox,
                        confidence=track.best_confidence,
                        class_name=track.class_name,
                        text=track.best_text,
                        track_id=track.track_id,
                    )
                )

        for track_id in expired:
            self._tracks.pop(track_id, None)

        return outputs


__all__ = ["PlateTracker"]
