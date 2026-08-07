"""
Pitcher-name matching (FanDuel prop outcome descriptions -> the MLB
starting pitcher) and same-line Over/Under pairing for the
pitcher_strikeouts market.

Both are deliberately conservative: an ambiguous pitcher match, or a
market where Over and Under don't share the exact same strikeout line,
is treated as unusable rather than guessed at or patched together.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_pitcher_name(name: Optional[str]) -> str:
    """Strips accents, punctuation, and common suffixes (Jr./Sr./II/III),
    lowercases, and collapses whitespace -- so 'Jose Ramirez Jr.' and
    'Jose Ramirez, Jr' both normalize to 'jose ramirez'."""
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = without_accents.lower()
    cleaned = re.sub(r"[.,]", "", lowered)
    cleaned = cleaned.replace("-", " ")
    tokens = [t for t in cleaned.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def match_pitcher_name(target_name: str, candidate_names: list[str]) -> Optional[str]:
    """Returns the single candidate whose normalized form exactly matches
    the normalized target name, or None if there isn't exactly one --
    never guesses at a partial/fuzzy match."""
    target_normalized = normalize_pitcher_name(target_name)
    if not target_normalized:
        return None

    matches = [c for c in candidate_names if normalize_pitcher_name(c) == target_normalized]
    if len(matches) != 1:
        return None
    return matches[0]


@dataclass
class StrikeoutLine:
    line: float
    over_odds: int
    under_odds: int
    last_update: Optional[str]


def pair_over_under(outcomes: list[dict]) -> Optional[StrikeoutLine]:
    """`outcomes` is the list of {"name": "Over"|"Under", "point": float,
    "price": int, "last_update": str} entries for ONE pitcher's
    pitcher_strikeouts market. Over and Under MUST share the exact same
    `point` (line) -- if they don't, or if a side is entirely missing,
    the market is treated as incomplete and this returns None rather
    than fabricating the missing side or combining mismatched lines."""
    overs = [o for o in outcomes if (o.get("name") or "").lower() == "over"]
    unders = [o for o in outcomes if (o.get("name") or "").lower() == "under"]

    if len(overs) != 1 or len(unders) != 1:
        return None

    over, under = overs[0], unders[0]
    if over.get("point") is None or under.get("point") is None:
        return None
    if float(over["point"]) != float(under["point"]):
        return None
    if over.get("price") is None or under.get("price") is None:
        return None

    return StrikeoutLine(
        line=float(over["point"]),
        over_odds=int(over["price"]),
        under_odds=int(under["price"]),
        last_update=over.get("last_update") or under.get("last_update"),
    )
