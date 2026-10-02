"""Cost arithmetic for the weekly challenge. EVERY price here is an assumption supplied by the caller (no prices are built in and none were looked up):
the code turns measured token counts and seconds into dollars under a stated price card, so the result is a statement about the card, not about any vendor.

    card = PriceCard(input_per_m=3.0, output_per_m=15.0)               # dollars per million tokens: an ASSUMED hosted-model card
    per_1000 = card.per_1000(prompt_tokens=700, new_tokens=90)
    break_even_requests(one_off_cost=1.2, saving_per_request=0.003)    # requests until the one-off cost is paid back
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceCard:
    input_per_m: float
    output_per_m: float
    label: str = "assumed"

    def per_request(self, prompt_tokens: float, new_tokens: float) -> float:
        return (prompt_tokens * self.input_per_m + new_tokens * self.output_per_m) / 1e6

    def per_1000(self, prompt_tokens: float, new_tokens: float) -> float:
        return 1000 * self.per_request(prompt_tokens, new_tokens)


def compute_cost_per_1000(seconds_per_request: float, dollars_per_hour: float) -> float:
    """Self-hosting: the machine is billed by time, so a request costs its seconds times the machine's hourly price divided by 3600 (at full utilisation)."""
    return 1000 * seconds_per_request * dollars_per_hour / 3600


def training_cost(train_tokens: int, epochs: int, seconds: float, dollars_per_hour: float) -> float:
    """Wall-clock cost of a training run on a rented machine: seconds x the hourly price (the token counts are reported next to it, not used for the price)."""
    _ = (train_tokens, epochs)
    return seconds * dollars_per_hour / 3600


def break_even_requests(one_off_cost: float, saving_per_request: float) -> float:
    """How many requests until a one-off cost (data creation, training) is repaid by a per-request saving. Infinity if there is no saving."""
    return float("inf") if saving_per_request <= 0 else one_off_cost / saving_per_request
