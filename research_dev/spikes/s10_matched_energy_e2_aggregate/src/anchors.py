#!/usr/bin/env python3
"""Typed external-anchor capability for E2A. Standard library only.

This module is the reason E2A cannot emit a physical label today, so it is worth
reading before anything else.

THE PROPERTY THE AGGREGATE NEEDS
--------------------------------
SUM_ALL_PAIRS_V1 sums EVERY planned pair. That rule is only worth something if
"every planned pair" was fixed before the results were seen. Otherwise a producer
runs 20 pairs, keeps the best 8, and writes a plan declaring exactly those 8: the
arithmetic is honest and the cohort is not. That is E1's record-cherry-picking
lesson one level up -- freezing a STATISTIC is not the same as freezing WHICH
RECORDS it is taken from.

So a plan commitment needs two independent properties:

    P1 PRECEDENCE   the plan digest existed before the first attempt ran.
    P2 EXCLUSIVITY  exactly ONE plan was committed for this experiment identity.

These do not covary, and conflating them is the trap. An RFC3161 timestamp is a
genuine third-party attestation and it buys P1 completely. It buys NOTHING of P2:
a TSA is a responder, not a log. It does not publish, enumerate, or cross-link the
tokens it issues, and there is no query a reviewer can run for "every token this
requester ever obtained". A producer can timestamp 32 candidate plans before
running anything, run everything, and reveal the one token whose plan fits the
results. Every check passes: the token is authentic, the imprint matches, the
genTime genuinely precedes the runs.

    An RFC3161 token is a lower bound on a plan's AGE.
    It is never an upper bound on a plan's COUNT.

Only an ENUMERABLE commitment closes P2 -- for example, a transparency-log query
that proves completeness for an externally assigned experiment identity, or
third-party pre-registration. A producer-selected inclusion proof is not enough.
See ANCHOR_AUDIT.md section 2.

WHY THIS IS TYPED AND NOT NAMED
-------------------------------
E2's instrument lesson, applied to integrity: a free-form `instrument_label`
grants nothing; a closed `instrument_kind` -> allowed-scope map is what actually
blocks promotion. The same shape here. `anchor_kind` is a closed enum, the
properties are frozen maps, and `anchor_property` is DERIVED from them -- never a
field a producer can supply. A record cannot talk its way into independence by
setting verifier_name to "freetsa.org".
"""

from __future__ import annotations

# Closed enum. A kind outside this set is E_ANCHOR_KIND, never a soft warning.
ANCHOR_KINDS = frozenset({
    "RFC3161_TSA",
    "TRANSPARENCY_LOG_INCLUSION",
    "TPM_NV_QUOTE",
    "SELF_SIGNED_TSA",
    "LOCAL_HMAC",
    "LOCAL_GPG_SIGNATURE",
    "GIT_COMMIT_LOCAL",
    "LOCAL_CLOCK_ASSERTION",
    "BMC_EVENT_LOG",
    "SYNTHETIC_ANCHOR",
})

# Is the attesting key held by a party the experiment cannot compel?
# Operational test: can the experiment uid, or a root the experimenter can become,
# or a force-push to a fork they own, produce a valid receipt for an arbitrary
# digest at an arbitrary time? If yes, not independent.
ANCHOR_INDEPENDENT = {
    # Third-party signing key, third-party clock. The experimenter cannot forge it.
    "RFC3161_TSA": True,
    # A third-party log can be independent even when the submitted identity is
    # insufficient to prove exclusivity.
    "TRANSPARENCY_LOG_INCLUSION": True,
    # Cryptographically sound, custodially ours: /dev/tpmrm0 is tss:tss and tss is
    # grantable by the root the experimenter can become. Same admin domain.
    "TPM_NV_QUOTE": False,
    # `openssl ts -reply -signer mine.pem` is a twenty-line offline TSA. The stock
    # openssl.cnf on this host already points [tsa] at a local ./demoCA.
    "SELF_SIGNED_TSA": False,
    # The key is the experiment's.
    "LOCAL_HMAC": False,
    # Authorship at best; the time is self-asserted. No key is even configured here.
    "LOCAL_GPG_SIGNATURE": False,
    # GIT_COMMITTER_DATE is settable and a personal fork is force-pushable.
    "GIT_COMMIT_LOCAL": False,
    # A string.
    "LOCAL_CLOCK_ASSERTION": False,
    # No BMC on this host; would be the same admin domain regardless.
    "BMC_EVENT_LOG": False,
    # Mechanics only. Must poison any MEASURED provenance.
    "SYNTHETIC_ANCHOR": False,
}

# What the receipt actually proves about time.
#   TIME_UPPER_BOUND: a third party asserts the digest existed before a wall time.
#   ORDER_ONLY:       a counter/tick proves sequence, not wall time.
#   NOTHING:          self-asserted.
ANCHOR_PROVES = {
    "RFC3161_TSA": "TIME_UPPER_BOUND",
    "TRANSPARENCY_LOG_INCLUSION": "TIME_UPPER_BOUND",
    "TPM_NV_QUOTE": "ORDER_ONLY",
    "SELF_SIGNED_TSA": "NOTHING",
    "LOCAL_HMAC": "NOTHING",
    "LOCAL_GPG_SIGNATURE": "NOTHING",
    "GIT_COMMIT_LOCAL": "NOTHING",
    "LOCAL_CLOCK_ASSERTION": "NOTHING",
    "BMC_EVENT_LOG": "NOTHING",
    "SYNTHETIC_ANCHOR": "NOTHING",
}

# THE LOAD-BEARING MAP. Can a reviewer enumerate every commitment this identity
# ever made, and so detect anchor-many-reveal-one?
#
# No currently modelled receipt does. Note that RFC3161_TSA is True in
# ANCHOR_INDEPENDENT and False here: that pair of entries IS the finding. An
# anchor can be perfectly independent and still leave the aggregate wide open.
# A transparency-log inclusion proof is also false here: it proves membership of
# one submitted leaf, not completeness under an externally assigned experiment
# identity. Without that identity binding and a completeness query, a producer
# can submit 32 plans under 32 keys and reveal one.
ANCHOR_ENUMERABLE = {
    "RFC3161_TSA": False,
    # Inclusion of one producer-selected leaf is not enumeration. This remains
    # false until a verifier proves completeness under an externally assigned
    # experiment identity.
    "TRANSPARENCY_LOG_INCLUSION": False,
    "TPM_NV_QUOTE": False,
    "SELF_SIGNED_TSA": False,
    "LOCAL_HMAC": False,
    "LOCAL_GPG_SIGNATURE": False,
    "GIT_COMMIT_LOCAL": False,
    "LOCAL_CLOCK_ASSERTION": False,
    "BMC_EVENT_LOG": False,
    "SYNTHETIC_ANCHOR": False,
}

# Derived properties, in increasing strength.
PROPERTY_NONE = "NONE"
PROPERTY_ORDERING_ONLY = "ORDERING_ONLY"
PROPERTY_ORDERING_AND_ENUMERABLE = "ORDERING_AND_ENUMERABLE"

# What SUM_ALL_PAIRS_V1 demands before it will emit any physical label.
REQUIRED_PROPERTY = PROPERTY_ORDERING_AND_ENUMERABLE

# Pins for an RFC3161 verification, if one is ever attempted. These are code
# constants ON PURPOSE. Taking the CA through a `-CAfile` argument, a config file,
# or a fixture is the idiomatic and fatal mistake: the producer then chooses the
# authority, and `openssl ts -verify` against your own demoCA passes cleanly.
# They are None because nothing is provisioned on this host; None means the trust
# root check cannot pass, which is the correct fail-closed state.
FROZEN_TSA_ROOT_SHA256 = None
FROZEN_TSA_LEAF_SHA256 = None
FROZEN_TSA_POLICY_OID = None

# Which anchor kinds does E2A actually know how to VERIFY?
#
# None of them. This map is empty, and its emptiness is the honest statement: E2A
# v2 types anchor capability and does not implement a single cryptographic
# verifier. No RFC3161 token is parsed, no signature is checked, no inclusion
# proof is validated.
#
# It is separated from the capability maps because conflating them was a real
# hole. Earlier code granted TRANSPARENCY_LOG_INCLUSION the required property
# before checking a proof. The current capability map refuses it as
# unenumerable, and this registry remains a separate second gate: capability and
# cryptographic verification are distinct requirements.
#
# "I have no verifier for this kind" is a different sentence from "this kind is
# too weak", and both must be said out loud.
ANCHOR_VERIFIERS = {}

VERIFIED_TOKEN_FIELDS = frozenset({
    "message_imprint_sha256",
    "token_hash_algorithm",
    "anchor_time_utc_us",
    "serial_number_hex",
    "tsa_subject_dn",
    "tsa_leaf_cert_sha256",
    "tsa_policy_oid",
    "trust_root_id",
    "trust_root_sha256",
})


def check_verifier_available(anchor_kind, verifier_kind=None):
    """Refuse any mechanism pair E2A cannot actually execute."""
    if verifier_kind is None:
        matches = [key for key in ANCHOR_VERIFIERS if key[0] == anchor_kind]
    else:
        matches = [(anchor_kind, verifier_kind)] \
            if (anchor_kind, verifier_kind) in ANCHOR_VERIFIERS else []
    if not matches:
        raise AnchorError(
            "E_ANCHOR_NO_VERIFIER",
            f"E2A v2 implements no verifier for {anchor_kind}/"
            f"{verifier_kind or '*'}: it types anchor "
            f"capability but parses no token, checks no signature, and validates "
            f"no inclusion proof. A receipt whose cryptography nobody checked is "
            f"a JSON object making an assertion about itself. Building the "
            f"verifier is a prerequisite for any physical claim, and is not in "
            f"this checkpoint")
    return True


def verify_token(receipt, token):
    """Invoke the exact verifier selected by the typed mechanism pair."""
    key = (receipt["anchor_kind"], receipt["verifier_kind"])
    check_verifier_available(*key)
    verifier = ANCHOR_VERIFIERS[key]
    if not callable(verifier):
        raise AnchorError("E_ANCHOR_VERIFY",
                          f"verifier registry entry {key!r} is not callable")
    try:
        verified = verifier(token)
    except Exception as exc:
        raise AnchorError(
            "E_ANCHOR_VERIFY",
            f"{key[0]}/{key[1]} verifier failed: {type(exc).__name__}: {exc}") \
            from exc
    if not isinstance(verified, dict) or set(verified) != VERIFIED_TOKEN_FIELDS:
        raise AnchorError(
            "E_ANCHOR_VERIFY",
            f"{key[0]}/{key[1]} verifier did not return the complete authenticated "
            "claim set")
    return verified


def check_precedence_supported(_receipt, _verified):
    """Refuse until a witnessed launcher binds external time to each attempt."""
    raise AnchorError(
        "E_ANCHOR_PRECEDENCE_UNSUPPORTED",
        "the token's authenticated UTC time cannot be ordered against the local "
        "monotonic attempt ledger. A witnessed or attested acquisition launcher "
        "must verify the plan anchor before it enables each attempt")

# Custodians that disqualify a trust root regardless of anchor_kind.
DISQUALIFYING_CUSTODIANS = frozenset({
    "EXPERIMENT_USER", "HOST_ROOT", "UNKNOWN",
})


class AnchorError(ValueError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def anchor_property(anchor_kind):
    """Derive the property from the frozen maps. Never read from a record.

    This function is the only way to learn what an anchor is worth. There is
    deliberately no `anchor_property` field in any schema: a value a producer can
    write is a value a producer can lie about, and E2's whole instrument-typing
    result rests on that distinction.
    """
    if anchor_kind not in ANCHOR_KINDS:
        raise AnchorError("E_ANCHOR_KIND",
                          f"unknown anchor_kind {anchor_kind!r}; capability is "
                          f"typed and the enum is closed")
    if not ANCHOR_INDEPENDENT[anchor_kind]:
        return PROPERTY_NONE
    if ANCHOR_PROVES[anchor_kind] != "TIME_UPPER_BOUND":
        return PROPERTY_NONE
    if not ANCHOR_ENUMERABLE[anchor_kind]:
        return PROPERTY_ORDERING_ONLY
    return PROPERTY_ORDERING_AND_ENUMERABLE


def check_anchor_kind(anchor_kind):
    """Gate one anchor kind against what the aggregate requires.

    Raises AnchorError with the specific reason. There is no return value that
    means "good enough": either it raises or the kind carries the required
    property.
    """
    prop = anchor_property(anchor_kind)
    if not ANCHOR_INDEPENDENT[anchor_kind]:
        raise AnchorError(
            "E_ANCHOR_NOT_INDEPENDENT",
            f"anchor_kind {anchor_kind} is produced inside the experiment's own "
            f"trust domain (the experiment user, a root they can become, or a "
            f"repository they can force-push); a receipt the producer can mint "
            f"for any digest at any time commits them to nothing")
    if ANCHOR_PROVES[anchor_kind] != "TIME_UPPER_BOUND":
        raise AnchorError(
            "E_ANCHOR_ORDER",
            f"anchor_kind {anchor_kind} proves {ANCHOR_PROVES[anchor_kind]}, not "
            f"that the plan digest existed before a wall time; sequence without "
            f"a time bound cannot place the plan before the runs")
    if prop != REQUIRED_PROPERTY:
        raise AnchorError(
            "E_ANCHOR_UNENUMERABLE",
            f"anchor_kind {anchor_kind} provides {prop} but SUM_ALL_PAIRS_V1 "
            f"requires {REQUIRED_PROPERTY}. The available receipt attests that "
            f"ONE digest existed before a time, and nothing about how many other "
            f"digests were anchored under aliases or alongside it. A producer "
            f"may anchor N candidate plans before running, run everything, and "
            f"reveal only the plan that fits the results -- every per-record check "
            f"passes. Closing this needs an enumerable commitment (a transparency "
            f"log with inclusion proofs, or third-party pre-registration), not a "
            f"better timestamp")
    return prop


# Kinds whose trust anchor is an X.509 CA, and which the FROZEN_TSA_* pins
# describe. A transparency log's trust anchor is a log public key plus inclusion
# proofs -- a different mechanism entirely, which the TSA pins say nothing about.
#
# The distinction is load-bearing. In the historical implementation,
# TRANSPARENCY_LOG_INCLUSION was blocked solely by
# FROZEN_TSA_ROOT_SHA256 being None. A timestamp-authority
# constant was accidentally the last line of defence for a transparency-log
# anchor: two unrelated mechanisms sharing one guard, holding by luck. Provision
# a TSA root for the RFC3161 path (which ANCHOR_AUDIT.md calls ten minutes of
# work) and the log path would have silently opened.
CA_ROOTED_KINDS = frozenset({"RFC3161_TSA", "SELF_SIGNED_TSA"})


def check_trust_root(record):
    """The trust root must be pinned in code and held by an outside party.

    Runs BEFORE any signature verification. Verifying a token against a root the
    producer supplied is theatre: `openssl ts -verify -CAfile ./demoCA/cacert.pem`
    exits 0 against a TSA the producer runs.

    Applies only to CA-rooted kinds. Other kinds are not thereby excused: they
    are refused by check_verifier_available, which is a statement about them
    rather than a constant borrowed from a different mechanism.
    """
    if record.get("anchor_kind") not in CA_ROOTED_KINDS:
        return True
    custodian = record.get("trust_root_custodian")
    if custodian in DISQUALIFYING_CUSTODIANS:
        raise AnchorError(
            "E_ANCHOR_TRUST_ROOT",
            f"trust root custodian {custodian!r} is inside the experiment's own "
            f"administrative domain; an authority the producer controls attests "
            f"nothing")
    if FROZEN_TSA_ROOT_SHA256 is None:
        raise AnchorError(
            "E_ANCHOR_TRUST_ROOT",
            "no timestamp-authority trust root is pinned in E2A's frozen "
            "constants, and no TSA root is provisioned on this host (see "
            "ANCHOR_AUDIT.md section 1.2); a token cannot be verified against a "
            "root that does not exist, and accepting a caller-supplied root would "
            "let the producer be its own authority")
    if record.get("trust_root_sha256") != FROZEN_TSA_ROOT_SHA256:
        raise AnchorError(
            "E_ANCHOR_TRUST_ROOT",
            "trust_root_sha256 does not match the pinned root")
    return True


def describe_host_capability():
    """What this host can actually anchor with, as measured in ANCHOR_AUDIT.md.

    Present so the audit's conclusion is executable rather than prose. Returns the
    best property reachable here.
    """
    # Measured 2026-07-15. RFC3161 is reachable and would verify once a CA is
    # pinned; nothing else is even that close. See ANCHOR_AUDIT.md.
    available = ["RFC3161_TSA"]
    best = PROPERTY_NONE
    order = {PROPERTY_NONE: 0, PROPERTY_ORDERING_ONLY: 1,
             PROPERTY_ORDERING_AND_ENUMERABLE: 2}
    for kind in available:
        prop = anchor_property(kind)
        if order[prop] > order[best]:
            best = prop
    return {"available_kinds": available, "best_property": best,
            "required_property": REQUIRED_PROPERTY,
            "sufficient": best == REQUIRED_PROPERTY}
