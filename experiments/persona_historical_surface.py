"""Historical v2 compatibility recovered from 5b02fdc16b25f90f95df4814907b672db8d24e1b.

Mapping and transforms originate in experiments/persona_surface_derivation.py;
authentication adapts experiments/persona_interference_schedule.py. This is a
recovery candidate, not proof of the unavailable historical execution source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


DERIVATION_VERSION = "persona-surface-derivation.v2"
DERIVATION_ALGORITHM = "category_block_permutation"
SURFACE_ASSIGNMENT_SEEDS = {"a": 137, "b": 911}
HISTORICAL_MODEL_IDENTITY = "deterministic-derived-surface"
SOURCE_ARTIFACTS = (
    "candidate_updates.jsonl",
    "events.jsonl",
    "examples.jsonl",
    "facts.jsonl",
    "queries.jsonl",
    "source_documents.jsonl",
)
_REQUIRED_ARTIFACTS = {*SOURCE_ARTIFACTS, "dialogue.jsonl", "requests.jsonl", "raw_responses.jsonl"}

CATEGORY_SOURCE_PHRASES = {
    "tea": ("cedar tea", "mint tea", "oolong tea"),
    "seating": ("window seating", "aisle seating"),
    "food": ("vegetable ramen", "mushroom risotto"),
    "delivery": ("evening delivery", "weekend delivery"),
    "receipt": ("digital receipts",),
    "private_lineage": ("cedar glass",),
}

SEATING_ROLE_VOCABULARIES = {
    "positive": (
        "window-side seating", "view-facing seating", "outward-facing seating",
        "perimeter seating", "outer-edge seating", "wall-adjacent seating",
        "light-facing seating", "exterior-side seating", "edge-position seating",
        "boundary-side seating", "side-wall seating", "natural-light seating",
        "view-oriented seating", "outside-edge seating", "window-adjacent seating",
        "wall-side seating",
    ),
    "avoided": (
        "aisle-side seating", "passage-side seating", "walkway-side seating",
        "corridor-side seating", "traffic-side seating", "foot-traffic seating",
        "cross-traffic seating", "entry-path seating", "exit-path seating",
        "access-lane seating", "circulation-path seating", "open-passage seating",
        "main-walkway seating", "through-route seating", "doorway-side seating",
        "central-passage seating",
    ),
}

CATEGORY_VOCABULARIES = {
    "tea": (
        "assam tea", "sencha tea", "jasmine tea", "chamomile tea",
        "rooibos tea", "darjeeling tea", "earl grey tea", "genmaicha tea",
        "hojicha tea", "matcha tea", "lapsang tea", "pu-erh tea",
        "white peony tea", "lemon ginger tea", "hibiscus tea", "barley tea",
        "cinnamon tea", "rosehip tea", "nettle tea", "sage tea",
        "thyme tea", "fennel tea", "anise tea", "cardamom tea",
        "lavender tea", "rose tea", "peach tea", "apricot tea",
        "plum tea", "quince tea", "orange blossom tea", "lemongrass tea",
        "tulsi tea", "osmanthus tea", "buckwheat tea", "chrysanthemum tea",
        "honeybush tea", "verbena tea", "elderflower tea", "raspberry leaf tea",
        "blackcurrant tea", "vanilla tea", "coconut tea", "ginger turmeric tea",
        "apple spice tea", "blueberry tea", "magnolia tea", "saffron tea",
    ),
    "seating": (
        *SEATING_ROLE_VOCABULARIES["positive"],
        *SEATING_ROLE_VOCABULARIES["avoided"],
    ),
    "food": (
        "spinach dumplings", "lentil curry", "sesame noodles", "pumpkin ravioli",
        "chickpea stew", "tomato gnocchi", "eggplant lasagna", "coconut rice",
        "herb polenta", "bean enchiladas", "miso udon", "pesto orzo",
        "saffron couscous", "sweet potato tacos", "broccoli soba", "corn chowder",
        "zucchini fritters", "red pepper pasta", "black bean soup", "lemon risotto",
        "ginger noodles", "butternut ravioli", "sesame tofu", "mushroom barley",
        "tomato farro", "spinach curry", "lentil pilaf", "vegetable paella",
        "chickpea tagine", "pumpkin soup", "herb dumplings", "coconut noodles",
    ),
    "delivery": (
        "dawn delivery", "sunrise delivery", "breakfast delivery", "midday delivery",
        "afternoon delivery", "sunset delivery", "twilight delivery", "night delivery",
        "Monday delivery", "Tuesday delivery", "Wednesday delivery", "Thursday delivery",
        "Friday delivery", "Saturday delivery", "Sunday delivery", "holiday delivery",
        "courier delivery", "porch delivery", "lobby delivery", "reception delivery",
        "office delivery", "home delivery", "curbside delivery", "doorstep delivery",
        "scheduled delivery", "express delivery", "standard delivery", "priority delivery",
        "same-day delivery", "next-day delivery", "fortnightly delivery", "monthly delivery",
    ),
    "receipt": (
        "printed receipts", "email receipts", "text receipts", "itemized receipts",
        "compact receipts", "full-page receipts", "mailed receipts", "app receipts",
        "thermal receipts", "carbon-copy receipts", "PDF receipts", "paperless receipts",
        "register receipts", "invoice receipts", "summary receipts", "detailed receipts",
    ),
    "private_lineage": (
        "amber compass", "silver lantern", "marble key", "willow token",
        "copper feather", "indigo ribbon", "maple pendant", "opal marker",
        "linen bookmark", "bronze charm", "violet seal", "crystal button",
        "walnut emblem", "scarlet pebble", "ivory medallion", "satin badge",
    ),
}

# Specific phrases precede descriptors so "cedar glass" cannot be consumed by
# the bare "cedar" tea rule. Descriptor rules cover lexical remnants observed
# in the authenticated parent prose after canonical phrase substitution.
ROLE_VARIANT_RULES = (
    ("cedar glass", r"\bcedar[-\s]+glass\b", "full"),
    ("cedar glass", r"\b(?:cedar|glass)[-\s]+keepsake\b", "full"),
    ("cedar tea", r"\bcedar\s+tea\b", "full"),
    ("mint tea", r"\bmint\s+tea\b", "full"),
    ("oolong tea", r"\boolong\s+tea\b", "full"),
    ("window seating", r"\bwindow\s+seating\b", "full"),
    ("window seating", r"\bwindow\s+seat\b", "seat"),
    ("window seating", r"\bwindow\s+seats\b", "seats"),
    ("window seating", r"\bwindow\s+spot\b", "spot"),
    ("window seating", r"\bwindow\s+spots\b", "spots"),
    ("aisle seating", r"\baisle\s+seating\b", "full"),
    ("aisle seating", r"\baisle\s+seat\b", "seat"),
    ("aisle seating", r"\baisle\s+seats\b", "seats"),
    ("aisle seating", r"\baisle\s+spot\b", "spot"),
    ("aisle seating", r"\baisle\s+spots\b", "spots"),
    ("vegetable ramen", r"\bvegetable\s+ramen\b", "full"),
    ("mushroom risotto", r"\bmushroom\s+risotto\b", "full"),
    ("evening delivery", r"\bevening\s+delivery\b", "full"),
    ("weekend delivery", r"\bweekend\s+delivery\b", "full"),
    ("digital receipts", r"\bdigital\s+receipts?\b", "full"),
    ("weekend delivery", r"\bweekend\b(?=\s+date\b)", "full"),
    (
        "evening delivery",
        r"\bevening\b(?=\s+(?:option|choice|selection|preference|timing|schedule)\b)",
        "descriptor",
    ),
    (
        "weekend delivery",
        r"\bweekend\b(?=\s+(?:option|choice|selection|preference|timing|schedule)\b)",
        "descriptor",
    ),
    ("cedar tea", r"\bcedar(?:-flavored)?\b(?!\s+glass\b)", "full"),
    ("cedar tea", r"\bwoodsy\b(?=\s+flavou?r\b)", "full"),
    ("mint tea", r"\bminty?\b", "full"),
    ("oolong tea", r"\boolong\b", "full"),
)

_COMPILED_ROLE_VARIANT_RULES = tuple(
    (source, re.compile(pattern, re.IGNORECASE), mode)
    for source, pattern, mode in ROLE_VARIANT_RULES
)


def _sha256_bytes(payload: bytes) -> str:
    """Return a byte payload's SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _stable_hash(value: Any) -> str:
    """Hash one JSON value with canonical serialization."""
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize stable object-valued JSONL."""
    return "".join(
        json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _load_jsonl(payload: bytes, name: str) -> list[dict[str, Any]]:
    """Load object-valued JSONL and name malformed rows."""
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"parent artifact {name} is not UTF-8: {error}") from error
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in parent {name} line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"parent {name} line {line_number} must be an object")
        rows.append(row)
    return rows


def _digest_int(seed: int, category: str) -> int:
    """Mix one integer seed and category without process-randomized hashing."""
    payload = f"{seed}:{category}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_surface_mapping(history_ids: Sequence[str], seed: int) -> list[dict[str, str]]:
    """Assign category terms by seed-rotated role blocks and history permutations."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if any(not isinstance(value, str) or not value for value in history_ids):
        raise ValueError("history identifiers must be non-empty strings")
    if len(set(history_ids)) != len(history_ids):
        raise ValueError("history identifiers must be unique")
    mapping = []
    history_count = len(history_ids)
    for category, sources in CATEGORY_SOURCE_PHRASES.items():
        vocabulary = CATEGORY_VOCABULARIES[category]
        required = history_count * len(sources)
        if len(vocabulary) != required:
            raise ValueError(
                f"category {category} requires exactly {required} terms for {history_count} histories, "
                f"got {len(vocabulary)}"
            )
        block_offset = (
            0
            if category == "seating"
            else _digest_int(seed, category) % len(sources)
        )
        for role_index, source in enumerate(sources):
            block_index = (role_index + block_offset) % len(sources)
            terms = list(
                vocabulary[block_index * history_count : (block_index + 1) * history_count]
            )
            random.Random(
                f"{DERIVATION_VERSION}:{seed}:{category}:{source}:histories"
            ).shuffle(terms)
            for history_id, target in zip(history_ids, terms, strict=True):
                mapping.append(
                    {
                        "category": category,
                        "history_id": history_id,
                        "source_phrase": source,
                        "target_phrase": target,
                    }
                )
    return sorted(
        mapping,
        key=lambda row: (row["history_id"], row["category"], row["source_phrase"]),
    )


def _replacement_index(mapping: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Index complete mappings by history and source phrase."""
    result: dict[str, dict[str, str]] = {}
    for row in mapping:
        history_mapping = result.setdefault(row["history_id"], {})
        if row["source_phrase"] in history_mapping:
            raise ValueError(
                f"duplicate source phrase {row['source_phrase']!r} for {row['history_id']}"
            )
        history_mapping[row["source_phrase"]] = row["target_phrase"]
    return result


def _preserve_capitalization(matched: str, replacement: str) -> str:
    """Apply all-caps or initial capitalization from a matched role phrase."""
    letters = "".join(character for character in matched if character.isalpha())
    if letters and letters.isupper():
        return replacement.upper()
    if matched and matched[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _role_replacement(target: str, mode: str) -> str:
    """Return a full category term or its grammatical role descriptor."""
    if mode == "full":
        return target
    if mode == "descriptor" and target.casefold().endswith(" delivery"):
        return target[: -len(" delivery")]
    if mode in {"seat", "seats", "spot", "spots"} and target.casefold().endswith(
        " seating"
    ):
        return target[: -len(" seating")] + f" {mode}"
    raise ValueError(f"unsupported role replacement mode {mode!r} for {target!r}")


def _repair_indefinite_articles(text: str, targets: Sequence[str]) -> str:
    """Repair a/an when a seating variant changes the following initial sound."""
    for target in sorted(set(targets), key=lambda value: (-len(value), value)):
        expected = "an" if target[0].casefold() in "aeiou" else "a"
        pattern = re.compile(rf"\b(a|an)(?=\s+{re.escape(target)}\b)", re.IGNORECASE)

        def replace_article(match: re.Match[str]) -> str:
            return _preserve_capitalization(match.group(0), expected)

        text = pattern.sub(replace_article, text)
    return text


def _replace_visible_text(text: str, replacements: Mapping[str, str]) -> str:
    """Replace canonical phrases and semantic role variants case-insensitively."""
    for source, pattern, mode in _COMPILED_ROLE_VARIANT_RULES:
        target = replacements.get(source)
        if target is None:
            continue
        replacement = _role_replacement(target, mode)
        text = pattern.sub(
            lambda match: _preserve_capitalization(match.group(0), replacement),
            text,
        )
    replacement_forms = list(replacements.values())
    replacement_forms.extend(
        _role_replacement(replacements[source], mode)
        for source, _, mode in _COMPILED_ROLE_VARIANT_RULES
        if source in replacements and mode in {"seat", "seats", "spot", "spots"}
    )
    text = _repair_indefinite_articles(text, replacement_forms)
    return text


def _stale_role_variants(text: str, replacements: Mapping[str, str]) -> list[str]:
    """Return every old same-history role variant still present in visible text."""
    stale = []
    for source, pattern, _ in _COMPILED_ROLE_VARIANT_RULES:
        if source not in replacements:
            continue
        stale.extend(
            f"{source}:{match.group(0)}" for match in pattern.finditer(text)
        )
    return sorted(set(stale), key=str.casefold)


def _validate_visible_transform(
    parent_text: str,
    transformed_text: str,
    replacements: Mapping[str, str],
    context: str,
) -> None:
    """Reject stale variants, missing targets, and source/target contradictions."""
    stale = _stale_role_variants(transformed_text, replacements)
    for source, target in replacements.items():
        source_stale = [
            value
            for value in stale
            if value.casefold().startswith(source.casefold() + ":")
        ]
        if source_stale and target.casefold() in transformed_text.casefold():
            raise ValueError(
                f"{context} contains a surface contradiction for {source!r}: {source_stale}"
            )
    if stale:
        raise ValueError(f"{context} retains stale role variants: {stale}")
    for source, target in replacements.items():
        matched_modes = set()
        for rule_source, pattern, mode in _COMPILED_ROLE_VARIANT_RULES:
            if rule_source == source and pattern.search(parent_text):
                matched_modes.add(mode)
        expected_targets = {
            _role_replacement(target, mode).casefold() for mode in matched_modes
        }
        missing_targets = sorted(
            expected
            for expected in expected_targets
            if expected not in transformed_text.casefold()
        )
        if missing_targets:
            raise ValueError(
                f"{context} lost replacement targets {missing_targets} for visible source {source!r}"
            )


def transform_surface_artifacts(
    artifacts: Mapping[str, bytes], mapping: Sequence[Mapping[str, str]]
) -> dict[str, bytes]:
    """Apply the canonical visible-surface mapping while preserving latent fields."""
    replacements = _replacement_index(mapping)
    parent_events = _load_jsonl(artifacts["events.jsonl"], "events.jsonl")
    parent_dialogue = _load_jsonl(artifacts["dialogue.jsonl"], "dialogue.jsonl")
    parent_queries = _load_jsonl(artifacts["queries.jsonl"], "queries.jsonl")
    if len(parent_dialogue) != len(parent_events):
        raise ValueError("parent dialogue and events must have identical row counts")

    events = []
    for event in parent_events:
        history_id = str(event.get("history_id", ""))
        if history_id not in replacements:
            raise ValueError(f"event references unknown history {history_id!r}")
        model_text = event.get("model_text")
        surface_object = event.get("surface_object")
        if not isinstance(model_text, str) or not isinstance(surface_object, str):
            raise ValueError(f"event {event.get('event_id')} lacks model-visible surfaces")
        transformed = dict(event)
        transformed["model_text"] = _replace_visible_text(
            model_text, replacements[history_id]
        )
        transformed["surface_object"] = _replace_visible_text(
            surface_object, replacements[history_id]
        )
        _validate_visible_transform(
            model_text,
            transformed["model_text"],
            replacements[history_id],
            f"event {event.get('event_id')} model_text",
        )
        events.append(transformed)

    dialogue = []
    for index, (row, parent_event, event) in enumerate(
        zip(parent_dialogue, parent_events, events, strict=True), start=1
    ):
        if row.get("text") != parent_event.get("model_text"):
            raise ValueError(f"parent dialogue row {index} does not match its event model_text")
        transformed = dict(row)
        transformed["text"] = event["model_text"]
        dialogue.append(transformed)

    queries = []
    for query in parent_queries:
        history_id = str(query.get("history_id", ""))
        if history_id not in replacements:
            raise ValueError(f"query references unknown history {history_id!r}")
        transformed = dict(query)
        for field in ("query_text", "surface_query_text", "surface_gold"):
            value = query.get(field)
            if not isinstance(value, str):
                raise ValueError(f"query {query.get('query_id')} lacks string field {field}")
            transformed[field] = _replace_visible_text(value, replacements[history_id])
            _validate_visible_transform(
                value,
                transformed[field],
                replacements[history_id],
                f"query {query.get('query_id')} {field}",
            )
        queries.append(transformed)

    output = dict(artifacts)
    output["events.jsonl"] = _jsonl_bytes(events)
    output["dialogue.jsonl"] = _jsonl_bytes(dialogue)
    output["queries.jsonl"] = _jsonl_bytes(queries)
    return output


def _validate_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _safe_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name not in {".", ".."}
        and not any(character in name for character in '/\\:<>"|?*')
        and not any(ord(character) < 32 for character in name)
        and not name.endswith((".", " "))
        and re.fullmatch(
            r"(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?",
            name,
            re.IGNORECASE,
        ) is None
    )


def _read_artifact(directory: Path, name: str) -> bytes:
    if not _safe_name(name):
        raise ValueError(f"invalid artifact name: {name!r}")
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError(f"artifact must be a regular file inside its corpus: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read corpus artifact {path}: {error}") from error


def _load_manifest(directory: Path, expected: str, label: str) -> dict[str, Any]:
    _validate_digest(expected, f"{label} manifest hash")
    payload = _read_artifact(directory, "generation_manifest.json")
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"{label} generation manifest hash mismatch: expected {expected}, got {actual}")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label} generation manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError(f"{label} generation manifest is not completed")
    return manifest


def _artifact_hashes(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not _REQUIRED_ARTIFACTS.issubset(value):
        raise ValueError(f"{label} manifest omits required artifact hashes")
    for name, digest in value.items():
        if not _safe_name(name) or name == "generation_manifest.json":
            raise ValueError(f"invalid {label} artifact name: {name!r}")
        _validate_digest(digest, f"{label} artifact {name}")
    return value


def _authenticated_artifacts(directory: Path, hashes: Mapping[str, str], label: str) -> dict[str, bytes]:
    artifacts = {}
    for name, expected in sorted(hashes.items()):
        payload = _read_artifact(directory, name)
        actual = _sha256_bytes(payload)
        if actual != expected:
            raise ValueError(f"{label} artifact hash mismatch for {name}: expected {expected}, got {actual}")
        artifacts[name] = payload
    return artifacts


def authenticate_historical_dataset(
    dataset_dir: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any] | None]:
    dataset_dir = Path(dataset_dir)
    manifest = _load_manifest(dataset_dir, expected_manifest_sha256, "historical")
    if manifest.get("model_identity") != HISTORICAL_MODEL_IDENTITY or manifest.get("method") is not None:
        raise ValueError("generation manifest is not a historical deterministic-derived-surface corpus")
    derivation = manifest.get("derivation")
    if (
        not isinstance(derivation, dict)
        or derivation.get("algorithm") != DERIVATION_ALGORITHM
        or derivation.get("version") != DERIVATION_VERSION
        or isinstance(derivation.get("seed"), bool)
        or not isinstance(derivation.get("seed"), int)
    ):
        raise ValueError("historical generation manifest has an invalid derivation contract")
    recorded = _artifact_hashes(manifest.get("artifact_sha256"), "child")
    parent = manifest.get("parent")
    if not isinstance(parent, dict):
        raise ValueError("historical generation manifest lacks parent provenance")
    parent_hash = parent.get("generation_manifest_sha256")
    _validate_digest(parent_hash, "parent manifest hash")
    parent_hashes = _artifact_hashes(parent.get("artifact_sha256"), "parent")
    if set(recorded) != set(parent_hashes):
        raise ValueError("historical parent and child artifact hash sets differ")
    parent_name = parent.get("corpus_name")
    if not _safe_name(parent_name):
        raise ValueError("historical generation manifest has an invalid parent corpus name")
    resolved_dataset = dataset_dir.resolve()
    parent_candidate = resolved_dataset.parent / parent_name
    parent_dir = parent_candidate.resolve()
    if parent_candidate.is_symlink() or parent_dir.parent != resolved_dataset.parent or parent_dir == resolved_dataset:
        raise ValueError("historical generation parent must resolve to a sibling corpus")
    parent_manifest = _load_manifest(parent_dir, parent_hash, "actual parent")
    parent_model = parent_manifest.get("model_identity")
    if (
        not isinstance(parent_model, str)
        or "kimi" not in parent_model.casefold()
        or parent_manifest.get("method") is not None
        or parent_manifest.get("derivation") is not None
    ):
        raise ValueError("actual parent generation is not a direct Kimi corpus")
    if _artifact_hashes(parent_manifest.get("artifact_sha256"), "actual parent") != parent_hashes:
        raise ValueError("actual parent artifact hash manifest differs from derived provenance")
    expected_provenance = {
        key: parent_manifest.get(key)
        for key in (
            "generator_role", "model_identity", "prompt_schema_version",
            "provider_identity_assurance", "status",
        )
    }
    if parent.get("kimi_provenance") != expected_provenance:
        raise ValueError("copied Kimi parent provenance differs from actual parent")
    mapping = manifest.get("surface_mapping")
    if not isinstance(mapping, list) or _stable_hash(mapping) != manifest.get("surface_mapping_sha256"):
        raise ValueError("historical generation manifest surface mapping hash mismatch")
    expected_contract = [
        {"mode": mode, "pattern": pattern, "source_phrase": source}
        for source, pattern, mode in ROLE_VARIANT_RULES
    ]
    if (
        manifest.get("visible_replacement_contract") != expected_contract
        or manifest.get("visible_replacement_contract_sha256") != _stable_hash(ROLE_VARIANT_RULES)
    ):
        raise ValueError("historical generation manifest has an invalid visible replacement contract")
    parent_artifacts = _authenticated_artifacts(parent_dir, parent_hashes, "actual parent")
    events = _load_jsonl(parent_artifacts["events.jsonl"], "events.jsonl")
    history_ids = [event.get("history_id") for event in events]
    if not history_ids or any(not isinstance(value, str) or not value for value in history_ids):
        raise ValueError("authenticated parent events contain incomplete history identifiers")
    expected_mapping = build_surface_mapping(sorted(set(history_ids)), derivation["seed"])
    if mapping != expected_mapping:
        raise ValueError("historical surface mapping does not match its declared seed")
    child_artifacts = _authenticated_artifacts(dataset_dir, recorded, "child")
    expected_artifacts = transform_surface_artifacts(parent_artifacts, mapping)
    for name, expected in sorted(expected_artifacts.items()):
        actual = child_artifacts[name]
        if actual != expected:
            context = "query gold or causal contract" if name == "queries.jsonl" else "artifact"
            raise ValueError(
                f"historical {context} canonical transformation mismatch for {name}: "
                f"expected {_sha256_bytes(expected)}, got {_sha256_bytes(actual)}"
            )
    source_names = (*SOURCE_ARTIFACTS, "dialogue.jsonl")
    provenance = {
        "path": str(dataset_dir),
        "generation_model": HISTORICAL_MODEL_IDENTITY,
        "generation_manifest_sha256": expected_manifest_sha256,
        "artifact_sha256": {name: recorded[name] for name in source_names},
        "checkpoint_policy": "authenticated_parent_exact_indices",
        "derivation": dict(derivation),
        "parent_generation_manifest_sha256": parent_hash,
        "parent_path": str(parent_dir),
    }
    authenticated_parent = {
        "path": parent_dir,
        "generation_manifest_sha256": parent_hash,
        "artifact_bytes": parent_artifacts,
    }
    return provenance, {name: child_artifacts[name] for name in source_names}, authenticated_parent
