from __future__ import annotations

import re
from dataclasses import dataclass


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


@dataclass
class Correction:
    text: str
    raw_text: str
    corrected: bool
    distance: int
    similarity: float


class LexiconCorrector:
    """Conservative OCR correction against a small fixed lexicon.

    It only replaces a reading when both the edit distance and normalized
    similarity are acceptable.  Otherwise the raw OCR text is preserved.
    """

    def __init__(self, entries: list[str], max_distance: int = 2, min_similarity: float = 0.6) -> None:
        self.entries = [entry.strip().upper() for entry in entries if entry.strip()]
        self.max_distance = int(max_distance)
        self.min_similarity = float(min_similarity)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^A-Z0-9-]", "", text.upper())

    def correct(self, text: str) -> Correction:
        raw = text
        normalized = self._normalize(text)
        if not normalized or not self.entries:
            return Correction(text=text, raw_text=raw, corrected=False, distance=0, similarity=0.0)

        best_entry = normalized
        best_distance = 10**9
        best_similarity = 0.0
        for entry in self.entries:
            distance = levenshtein(normalized, entry)
            similarity = 1.0 - distance / max(len(normalized), len(entry), 1)
            if distance < best_distance or (distance == best_distance and similarity > best_similarity):
                best_entry = entry
                best_distance = distance
                best_similarity = similarity

        # Part number rows often contain a spurious leading character or a
        # trailing OCR artefact (for example "4LC3864NML-1Q").  If an "LC"
        # anchor is present, compare from that anchor with one extra error
        # allowed.  This is deliberately limited to the package part number.
        lc_entries = [entry for entry in self.entries if entry.startswith("LC")]
        lc_start = normalized.find("LC")
        if lc_entries and lc_start >= 0:
            anchored = normalized[lc_start:]
            for entry in lc_entries:
                distance = levenshtein(anchored, entry)
                similarity = 1.0 - distance / max(len(anchored), len(entry), 1)
                if distance <= self.max_distance + 1 and similarity >= max(0.7, self.min_similarity):
                    return Correction(
                        text=entry,
                        raw_text=raw,
                        corrected=entry != normalized,
                        distance=distance,
                        similarity=similarity,
                    )

        # A split line such as "JAPAN 2CB3" should not be forced into one of
        # the single-line lexicon entries.
        if len(normalized) > max(len(entry) for entry in self.entries) + 2:
            return Correction(
                text=text,
                raw_text=raw,
                corrected=False,
                distance=best_distance,
                similarity=best_similarity,
            )

        if best_distance <= self.max_distance and best_similarity >= self.min_similarity:
            return Correction(
                text=best_entry,
                raw_text=raw,
                corrected=best_entry != normalized,
                distance=best_distance,
                similarity=best_similarity,
            )
        return Correction(
            text=text,
            raw_text=raw,
            corrected=False,
            distance=best_distance,
            similarity=best_similarity,
        )
