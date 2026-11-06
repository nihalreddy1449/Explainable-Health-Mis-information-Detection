"""The two-tier system: route a piece of health text to the tier that can judge it.

    short claim (<= 60 words)  -> Tier 1: is the claim supported by biomedical evidence?
    full article (> 60 words)  -> Tier 2: does the article report health science responsibly?

The tiers answer different questions and are deliberately not interchangeable. The
60-word cut-off is a heuristic, not learned. On the project's data it routes 5,067 of
5,075 texts to the right tier: 8 of 2,956 Tier 1 claims are longer than 60 words
(the longest is 102), and no Tier 2 article is 60 words or shorter (the shortest
is 64). The 8 long claims are the router's known failure.

Usage:
    from pipeline import HealthMisinfoPipeline, route
    pipe = HealthMisinfoPipeline()
    out = pipe.analyse(text, explain=True)       # out["tier"], out["result"]
"""
from __future__ import annotations

ROUTER_MAX_WORDS = 60


def word_count(text: str) -> int:
    return len((text or "").split())


def route(text: str, max_words: int = ROUTER_MAX_WORDS) -> str:
    """'tier1' for a short claim, 'tier2' for an article."""
    return "tier1" if word_count(text) <= max_words else "tier2"


class HealthMisinfoPipeline:
    """Loads each tier's engine on first use, so routing a claim never pays for
    loading the article models and vice versa."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._t1 = self._t2 = None

    @property
    def tier1(self):
        if self._t1 is None:
            from tier1.engine import Tier1Engine
            self._t1 = Tier1Engine(device=self.device)
        return self._t1

    @property
    def tier2(self):
        if self._t2 is None:
            from tier2.engine import Tier2Engine
            self._t2 = Tier2Engine(device=self.device)
        return self._t2

    def analyse(self, text: str, tier: str | None = None, explain: bool = False) -> dict:
        """Route (unless `tier` forces one) and run. Returns the tier used, why, and
        that tier's own result dict."""
        text = (text or "").strip()
        n = word_count(text)
        chosen = tier or route(text)
        reason = ("forced" if tier else
                  f"{n} words {'<=' if chosen == 'tier1' else '>'} {ROUTER_MAX_WORDS}-word limit")
        if chosen == "tier1":
            result = self.tier1.verify(text, explain=explain)
        else:
            result = self.tier2.assess(text, explain=explain)
        return {"tier": chosen, "word_count": n, "reason": reason, "result": result}
