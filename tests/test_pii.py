"""Tests for common/pii.py: validators against known-good and known-bad numbers, every detector on positives AND on the look-alikes
it must leave alone, redaction modes, reversible pseudonyms, and the logging/LLM boundary helpers."""

from __future__ import annotations

import logging

import pytest

from common import pii


def found(text, **kw):
    return [(s.type, s.text) for s in pii.detect(text, **kw)]


def types(text, **kw):
    return {t for t, _ in found(text, **kw)}


# ----------------------------------------------------------------------------- validators


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111111",
        "5555555555554444",
        "378282246310005",
        "6011111111111117",
        "4012888888881881",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
    ],
)
def test_luhn_accepts_published_test_cards(number):
    assert pii.luhn_valid(number)


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111112",
        "5555555555554445",
        "1234567812345678",
        "4111",
        "",
        "41111111111111111111",
        "abcdefghijklmnop",
    ],
)
def test_luhn_rejects_wrong_check_digits_and_wrong_lengths(number):
    assert not pii.luhn_valid(number)


def test_every_single_digit_change_to_a_valid_card_breaks_the_check():
    good = "4111111111111111"
    for i in range(len(good)):
        for d in "0123456789":
            if d != good[i]:
                assert not pii.luhn_valid(good[:i] + d + good[i + 1 :]), (i, d)


@pytest.mark.parametrize(
    "iban",
    [
        "GB82 WEST 1234 5698 7654 32",
        "DE89370400440532013000",
        "FR1420041010050500013M02606",
        "NL91ABNA0417164300",
        "gb82west12345698765432",
    ],
)
def test_iban_accepts_published_examples(iban):
    assert pii.iban_valid(iban)


@pytest.mark.parametrize(
    "iban",
    [
        "GB82 WEST 1234 5698 7654 33",
        "GB00 WEST 1234 5698 7654 32",
        "DE8937040044053201300",
        "DE893704004405320130000",
        "XX00",
        "12345678901234567",
        "GB82WEST1234569876543",
    ],
)
def test_iban_rejects_bad_checksums_lengths_and_shapes(iban):
    assert not pii.iban_valid(iban)


@pytest.mark.parametrize(
    "area,group,serial,ok",
    [
        ("123", "45", "6789", True),
        ("000", "12", "3456", False),
        ("666", "12", "3456", False),
        ("900", "12", "3456", False),
        ("999", "12", "3456", False),
        ("123", "00", "6789", False),
        ("123", "45", "0000", False),
        ("899", "99", "9999", True),
    ],
)
def test_ssn_rules(area, group, serial, ok):
    assert pii.ssn_valid(area, group, serial) is ok


def test_entropy():
    assert pii.entropy("") == 0.0 and pii.entropy("aaaa") == 0.0
    assert pii.entropy("abcd") == pytest.approx(2.0)
    assert pii.entropy("a" * 50 + "b" * 50) == pytest.approx(1.0)
    assert pii.entropy("aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5") > 4.2


# ----------------------------------------------------------------------------- detection: positives


def test_emails_in_many_shapes():
    assert found("Mail alice@example.com or bob.smith+test@mail.co.uk now") == [
        ("EMAIL", "alice@example.com"),
        ("EMAIL", "bob.smith+test@mail.co.uk"),
    ]
    assert found("(first.last@sub.domain.example.org)") == [
        ("EMAIL", "first.last@sub.domain.example.org")
    ]
    assert found("x_y-z@a-b.io.") == [("EMAIL", "x_y-z@a-b.io")]


def test_cards_with_valid_checksums_are_found_in_all_separator_styles_and_only_those():
    assert found("Card 4111 1111 1111 1111 ok") == [("CREDIT_CARD", "4111 1111 1111 1111")]
    assert found("Card 4111-1111-1111-1111.") == [("CREDIT_CARD", "4111-1111-1111-1111")]
    assert found("4111111111111111") == [("CREDIT_CARD", "4111111111111111")]
    assert found("amex 3782 822463 10005") == [("CREDIT_CARD", "3782 822463 10005")]
    assert found("order 4111111111111112 shipped") == [], "fails Luhn: an order number, not a card"
    assert found("tracking 1234567812345678") == []
    assert found("0000 0000 0000 0000") == [], "valid checksum but one repeated digit"


def test_iban_ssn_and_context_ssn():
    assert found("pay to GB82 WEST 1234 5698 7654 32 please")[0] == (
        "IBAN",
        "GB82 WEST 1234 5698 7654 32",
    )
    assert ("IBAN", "DE89370400440532013000") in found("IBAN DE89370400440532013000.")
    assert found("SSN 123-45-6789.") == [("US_SSN", "123-45-6789")]
    assert found("my social security number is 123456789") == [("US_SSN", "123456789")]
    assert found("000-12-3456 and 666-45-6789 and 912-45-6789") == []


def test_phone_numbers_in_international_and_national_shapes():
    assert [t for t, _ in found("Call +1 (415) 555-0132")] == ["PHONE"]
    assert found("or 415-555-0132 or 415.555.0132") == [
        ("PHONE", "415-555-0132"),
        ("PHONE", "415.555.0132"),
    ]
    assert found("+84 90 123 4567") == [("PHONE", "+84 90 123 4567")]
    assert found("+442071838750") == [("PHONE", "+442071838750")]
    assert found("tel: 0412 345 678")[0][0] == "PHONE"
    assert found("whatsapp 0912345678")[0] == ("PHONE", "0912345678"), (
        "a bare number needs a context word"
    )


def test_ip_addresses_but_not_versions_or_invalid_octets():
    assert found("Server 192.168.1.10 up") == [("IP_ADDRESS", "192.168.1.10")]
    assert found("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 ok")[0][0] == "IP_ADDRESS"
    assert found("version 1.2.3.4 and v10.0.0.1 and 999.1.1.1 and 1.2.3.4.5") == []
    assert found("python 3.12.1.0") == []


def test_secrets_keys_tokens_and_credentials():
    assert types("key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789") == {"API_KEY"}
    assert types("AKIAABCDEFGHIJKLMNOP") == {"API_KEY"} and types(
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    ) == {"API_KEY"}
    assert types("Authorization: Bearer abcdefghijklmnopqrstuvwx1234567") == {"API_KEY"}
    assert found("password=hunter2hunter") == [("CREDENTIALS", "hunter2hunter")]
    assert found("api_key = 'AbCdEf123456'") == [("CREDENTIALS", "AbCdEf123456")]
    assert found("postgres://admin:s3cretPass@db.example.com:5432/prod") == [
        ("URL_CREDENTIALS", "admin:s3cretPass")
    ]
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert found(f"t={jwt}") == [("JWT", jwt)]
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n-----END RSA PRIVATE KEY-----"
    assert found(f"here:\n{pem}\nbye") == [("PRIVATE_KEY", pem)]
    assert found("-----BEGIN PRIVATE KEY-----\nabc")[0][0] == "PRIVATE_KEY", (
        "an unterminated key block still goes to the end"
    )


def test_high_entropy_tokens_are_flagged_with_low_confidence_and_hashes_and_words_are_not():
    token = "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5"
    assert found(f"token {token} end") == [("SECRET_TOKEN", token)]
    assert pii.detect(f"x {token}")[0].score == 0.6
    assert found("sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08") == [], (
        "a plain hex digest is not a secret"
    )
    assert (
        found("a" * 40) == []
        and found("supercalifragilisticexpialidocious_is_a_very_long_word") == []
    )


def test_heuristic_names_addresses_and_dates_of_birth():
    assert found("Hello, my name is Alice Johnson.") == [("PERSON", "Alice Johnson")]
    assert found("Dr. House will call") == [("PERSON", "House")] and found("Ms Jane Roe wrote") == [
        ("PERSON", "Jane Roe")
    ]
    assert found("I am Not Happy about this") == [], (
        "'I am' followed by capitalised words that are not a name"
    )
    assert found("I'm Looking forward to it") == []
    assert found("lives at 221 Baker Street, London") == [("ADDRESS", "221 Baker Street, London")]
    assert found("12 Elm Court") == [("ADDRESS", "12 Elm Court")]
    assert found("DOB: 1990-04-12") == [("DATE_OF_BIRTH", "1990-04-12")] and found(
        "born on March 3, 1985"
    ) == [("DATE_OF_BIRTH", "March 3, 1985")]
    assert found("date of birth 03/04/1985")[0][0] == "DATE_OF_BIRTH"
    assert found("the meeting is on 2024-05-17") == [], "a date is not a date of birth"


# ----------------------------------------------------------------------------- detection: look-alikes that must be left alone


@pytest.mark.parametrize(
    "text",
    [
        "Invoice INV-3001 for $49.00 was paid on 2024-05-17.",
        "Order 12345678 shipped; tracking 1Z999AA10123456784.",
        "Version 2.31.0 requires Python 3.12.",
        "Error 5003 means a temporary outage; retry after 30 seconds.",
        "The API allows 100 requests per minute (HTTP 429 on excess).",
        "Call the support line weekdays 9 to 17 UTC.",
        "Meeting 10:30-11:45, room 4B, floor 2.",
        "RF-0007 ($20.00) was issued; refund id 1234-5678.",
        "Coordinates 51.5074, -0.1278.",
        "SHA a94a8fe5ccb19ba61c4c0873d391e987982fbbd3.",
        "",
        "   ",
    ],
)
def test_ordinary_business_text_has_no_pii(text):
    assert found(text) == [], found(text)


# ----------------------------------------------------------------------------- overlaps, scores, filters


def test_overlapping_matches_go_to_the_higher_priority_type_and_spans_never_overlap():
    spans = pii.detect(
        "postgres://admin:s3cretPass@db.example.com/x and 4111 1111 1111 1111 and alice@example.com"
    )
    assert [s.type for s in spans] == ["URL_CREDENTIALS", "CREDIT_CARD", "EMAIL"]
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:], strict=False))
    assert (
        pii.detect("password=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789")[0].type
        == "API_KEY"
    ), "a vendor key outranks the generic assignment"


def test_min_score_and_type_filters():
    text = "My name is Alice Johnson and mail alice@example.com"
    assert {t for t, _ in found(text)} == {"PERSON", "EMAIL"}
    assert found(text, min_score=0.7) == [("EMAIL", "alice@example.com")]
    assert found(text, types={"PERSON"}) == [("PERSON", "Alice Johnson")]
    assert found(text, types={"PHONE"}) == [] and pii.detect("") == []


def test_spans_report_positions_that_slice_back_to_the_text():
    text = "xx alice@example.com yy 4111 1111 1111 1111 zz"
    for s in pii.detect(text):
        assert text[s.start : s.end] == s.text
    assert pii.Span("A", 0, 3, "abc", 1).overlaps(pii.Span("A", 2, 5, "cde", 1)) and not pii.Span(
        "A", 0, 3, "abc", 1
    ).overlaps(pii.Span("A", 3, 5, "de", 1))


# ----------------------------------------------------------------------------- redaction


def test_redaction_modes():
    t = "Mail alice@example.com, card 4111 1111 1111 1111."
    assert pii.redact(t) == "Mail [EMAIL], card [CREDIT_CARD]."
    assert pii.redact(t, mode="tag") == "Mail <EMAIL>, card <CREDIT_CARD>."
    h1, h2 = (
        pii.redact(t, mode="hash"),
        pii.redact("alice@example.com again alice@example.com", mode="hash"),
    )
    import re

    tags = re.findall(r"<EMAIL:([0-9a-f]{6})>", h1 + h2)
    assert len(tags) == 3 and len(set(tags)) == 1, (
        "the same value gets the same tag, so records can still be joined"
    )
    assert pii.redact("bob@example.com", mode="hash") != pii.redact(
        "alice@example.com", mode="hash"
    )
    assert pii.redact("alice@example.com", mode="hash", key=b"k1") != pii.redact(
        "alice@example.com", mode="hash", key=b"k2"
    )
    with pytest.raises(ValueError):
        pii.redact(t, mode="rot13")


def test_cards_can_keep_their_last_digits_for_support_staff():
    assert pii.redact("card 4111 1111 1111 1111 ok", keep_card_last=4) == "card ************1111 ok"
    assert pii.redact("mail a@b.example", keep_card_last=4) == "mail [EMAIL]"


def test_redaction_leaves_clean_text_byte_identical_and_is_idempotent():
    clean = "Refund RF-0007 ($20.00) was issued for INV-3001."
    assert pii.redact(clean) == clean
    once = pii.redact("alice@example.com and 4111 1111 1111 1111")
    assert pii.redact(once) == once


def test_redaction_keeps_the_text_around_the_spans():
    assert (
        pii.redact("a alice@example.com b 4111 1111 1111 1111 c") == "a [EMAIL] b [CREDIT_CARD] c"
    )
    assert (
        pii.redact("alice@example.com") == "[EMAIL]"
        and pii.redact("x\nalice@example.com\ny") == "x\n[EMAIL]\ny"
    )


# ----------------------------------------------------------------------------- pseudonyms


def test_pseudonyms_are_consistent_distinct_and_reversible():
    ps = pii.Pseudonymizer()
    safe = ps.pseudonymize(
        "Mail alice@example.com and bob@example.com, again alice@example.com; card 4111 1111 1111 1111."
    )
    assert safe == "Mail <EMAIL_1> and <EMAIL_2>, again <EMAIL_1>; card <CREDIT_CARD_1>."
    assert ps.vault == {
        "<EMAIL_1>": "alice@example.com",
        "<EMAIL_2>": "bob@example.com",
        "<CREDIT_CARD_1>": "4111 1111 1111 1111",
    }
    assert (
        ps.restore("Sent to <EMAIL_2> and <EMAIL_1> (card <CREDIT_CARD_1>)")
        == "Sent to bob@example.com and alice@example.com (card 4111 1111 1111 1111)"
    )


def test_restore_only_reverses_tokens_this_session_issued():
    ps = pii.Pseudonymizer()
    ps.pseudonymize("alice@example.com")
    assert (
        ps.restore("<EMAIL_1> and <EMAIL_9> and <FOO_1> and plain")
        == "alice@example.com and <EMAIL_9> and <FOO_1> and plain"
    )
    other = pii.Pseudonymizer()
    assert other.restore("<EMAIL_1>") == "<EMAIL_1>", "another session cannot read this one's vault"


def test_the_same_value_gets_the_same_token_across_turns_of_a_session():
    ps = pii.Pseudonymizer()
    a = ps.pseudonymize("I am alice@example.com")
    b = ps.pseudonymize("write to alice@example.com and carol@example.com")
    assert a.endswith("<EMAIL_1>") and b == "write to <EMAIL_1> and <EMAIL_2>"


def test_the_pseudonymized_text_contains_no_original_values():
    ps = pii.Pseudonymizer()
    original = "alice@example.com 4111 1111 1111 1111 +1 (415) 555-0132 123-45-6789"
    safe = ps.pseudonymize(original)
    assert (
        pii.leaks(safe, ["alice@example.com", "4111111111111111", "415-555-0132", "123-45-6789"])
        == []
    )


# ----------------------------------------------------------------------------- structured data, logging, the LLM boundary


def test_scrub_redacts_nested_strings_and_replaces_values_under_secret_keys():
    data = {
        "user": {
            "email": "alice@example.com",
            "password": "hunter2",
            "profile": ["call +1 (415) 555-0132", 42, None],
        },
        "Authorization": "Bearer abcdefghijklmnopqrstuvwx1234567",
        "n": 3,
        "token": "",
        "api_key": None,
    }
    out = pii.scrub(data)
    assert (
        out["user"]["email"] == "[EMAIL]"
        and out["user"]["password"] == "[REDACTED]"
        and out["Authorization"] == "[REDACTED]"
    )
    assert out["user"]["profile"] == ["call [PHONE]", 42, None] and out["n"] == 3
    assert out["token"] == "" and out["api_key"] is None, (
        "an empty secret field stays empty (nothing to protect, and the shape is kept)"
    )
    assert (
        pii.scrub(("a@b.example", "x")) == ("[EMAIL]", "x")
        and data["user"]["password"] == "hunter2"
    ), "the input is not modified"


def test_leaks_ignores_case_spaces_and_dashes():
    assert pii.leaks("card is 4111-1111-1111-1111 ok", ["4111111111111111", "other"]) == [
        "4111111111111111"
    ]
    assert (
        pii.leaks("Alice@Example.COM", ["alice@example.com"]) == ["alice@example.com"]
        and pii.leaks("clean", ["alice@example.com"]) == []
    )


def test_the_logging_filter_redacts_messages_and_arguments():
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("pii-test")
    log.setLevel(logging.INFO)
    h = Capture()
    h.addFilter(pii.redaction_filter())
    log.addHandler(h)
    try:
        log.info("user alice@example.com asked")
        log.info("card %s and count %d", "4111 1111 1111 1111", 3)
        log.info("no pii here")
    finally:
        log.removeHandler(h)
    assert records == ["user [EMAIL] asked", "card [CREDIT_CARD] and count 3", "no pii here"]


def test_the_shield_pseudonymises_the_prompt_and_restores_the_reply():
    seen = []

    def model(prompt, **kw):
        seen.append(prompt)
        return "Done: I emailed <EMAIL_1> about the charge on <CREDIT_CARD_1>."

    reply = pii.shield(model)("Refund alice@example.com for card 4111 1111 1111 1111")
    assert (
        seen == ["Refund <EMAIL_1> for card <CREDIT_CARD_1>"]
        and pii.leaks(seen[0], ["alice@example.com", "4111111111111111"]) == []
    )
    assert reply == "Done: I emailed alice@example.com about the charge on 4111 1111 1111 1111."


def test_the_shield_uses_a_fresh_vault_per_call_unless_given_one():
    calls = []
    wrapped = pii.shield(lambda p, **kw: calls.append(p) or "ok")
    wrapped("a@b.example")
    wrapped("c@d.example")
    assert calls == ["<EMAIL_1>", "<EMAIL_1>"]
    shared = pii.Pseudonymizer()
    w2 = pii.shield(lambda p, **kw: calls.append(p) or "ok", shared)
    w2("a@b.example")
    w2("c@d.example")
    assert calls[2:] == ["<EMAIL_1>", "<EMAIL_2>"]


# ----------------------------------------------------------------------------- boundaries found by the mutation checks


def luhn_complete(prefix: str) -> str:
    """Append the check digit that makes ``prefix`` Luhn-valid (an independent implementation: it works for any length)."""
    total = 0
    for i, ch in enumerate(reversed(prefix)):
        d = int(ch) * (
            2 if i % 2 == 0 else 1
        )  # the check digit will sit to the right, so this digit is at an odd position from it
        total += d - 9 if d > 9 else d
    return prefix + str((10 - total % 10) % 10)


def iban_with_check(country: str, bban: str) -> str:
    numeric = "".join(str(int(c, 36)) for c in bban + country + "00")
    return f"{country}{98 - int(numeric) % 97:02d}{bban}"


def test_a_luhn_valid_number_with_too_many_digits_is_not_a_card():
    twenty = luhn_complete("4" + "1234567890123456789")
    assert len(twenty) == 21 or len(twenty) == 20
    assert pii.luhn_valid(luhn_complete("411111111111111")) is True, (
        "the helper agrees with the module on a normal length"
    )
    assert pii.luhn_valid(twenty) is False
    assert found(f"ref {twenty} ok") == []


def test_the_iban_length_for_the_country_matters_even_when_the_checksum_is_right():
    right = iban_with_check("DE", "370400440532013000")  # 22 characters
    wrong = iban_with_check("DE", "37040044053201300")  # 21: valid mod 97, wrong length for Germany
    assert (
        len(right) == 22
        and pii.iban_valid(right)
        and len(wrong) == 21
        and not pii.iban_valid(wrong)
    )
    assert pii.iban_valid(iban_with_check("XK", "1234567890123456")) is True, (
        "a country without a table entry is only checksum-checked"
    )


def test_card_confidence_depends_on_a_known_brand_prefix():
    assert pii.detect("4111 1111 1111 1111")[0].score == 0.95
    unknown = luhn_complete("700000000000000")
    assert len(unknown) == 16 and pii.detect(unknown)[0].score == 0.6


def test_phone_confidence_and_digit_limits():
    assert (
        pii.detect("+1 (415) 555-0132")[0].score == 0.8
        and pii.detect("415-555-0132")[0].score == 0.65
    )
    assert (
        found("call 12345678901234567890") == []
        and found("phone 1111 1111") == []
        and found("tel 1212-1212") == []
    )
    assert found("tel 12345") == [], "too few digits"


def test_clock_times_are_not_ipv6_addresses():
    assert found("meeting at 10:30:45 sharp") == [] and found("ratio 1:2:3") == []


def test_a_span_exactly_at_min_score_is_kept():
    assert found("my name is Alice Johnson", min_score=0.6) == [("PERSON", "Alice Johnson")]
    assert found("my name is Alice Johnson", min_score=0.61) == []


def test_the_default_hash_key_is_random_per_process():
    assert len(pii._DEFAULT_KEY) == 16 and pii._DEFAULT_KEY != b"\x00" * 16


def test_a_low_entropy_alphanumeric_run_is_not_a_secret_and_a_hex_digest_cannot_reach_the_threshold():
    assert found("id " + "a1b2c3" * 7) == [], "long and mixed, but a repeating pattern"
    assert pii.entropy("0123456789abcdef" * 4) == pytest.approx(4.0) and 4.0 < 4.2


def test_the_default_hash_key_is_what_redact_uses_when_none_is_given(monkeypatch):
    monkeypatch.setattr(pii, "_DEFAULT_KEY", b"k")
    assert pii.redact("alice@example.com", mode="hash") == pii.redact(
        "alice@example.com", mode="hash", key=b"k"
    )
