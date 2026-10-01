"""Typed schemas for each document type, with semantic validators the JSON schema can't express."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DocType = Literal["invoice", "receipt", "offer_letter", "unknown"]
TOLERANCE = 0.011  # money comparisons: a rounding cent is fine


class Classification(BaseModel):
    doc_type: DocType = Field(description="unknown if it is none of the listed types")
    confidence: float = Field(ge=0, le=1, description="0 to 1")


class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: date = Field(description="ISO-8601, whatever format the document uses")
    due_date: date | None = None
    currency: Literal["USD", "EUR", "GBP"]
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float
    tax: float = Field(description="Tax amount (0 if none)")
    total: float

    @model_validator(mode="after")
    def arithmetic_must_add_up(self) -> Invoice:
        items = round(sum(i.quantity * i.unit_price for i in self.line_items), 2)
        if abs(items - self.subtotal) > TOLERANCE:
            raise ValueError(f"line items sum to {items} but subtotal is {self.subtotal}")
        if abs(self.subtotal + self.tax - self.total) > TOLERANCE:
            raise ValueError(f"subtotal {self.subtotal} + tax {self.tax} != total {self.total}")
        if self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date is before invoice_date")
        return self


class Receipt(BaseModel):
    merchant: str
    purchase_date: date
    total: float = Field(ge=0)
    payment_method: Literal["card", "cash", "other"]
    item_count: int = Field(ge=1)


class OfferLetter(BaseModel):
    candidate_name: str
    position: str
    start_date: date
    salary_amount: float = Field(gt=0)
    salary_period: Literal["yearly", "monthly"]
    currency: Literal["USD", "EUR", "GBP"]


SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "receipt": Receipt,
    "offer_letter": OfferLetter,
}


def sanity_date(d: date | None) -> None:
    if d and not (2015 <= d.year <= 2035):
        raise ValueError(f"date {d} is implausible")
