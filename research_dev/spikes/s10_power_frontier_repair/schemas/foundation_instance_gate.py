#!/usr/bin/env python3
"""Dependency-free gate for the checked-in foundation instance schema.

The temporal oracle and checker already enforce the cross-field invariants that
JSON Schema cannot express. This gate covers the structural schema first. It
supports exactly the keywords used by instance.schema.json and fails closed if
the schema starts using a keyword this implementation does not understand.
"""

from __future__ import annotations

import functools
import json
import pathlib
import re


SCHEMA_PATH = pathlib.Path(__file__).with_name("instance.schema.json")

_SUPPORTED = {
    "$schema", "$id", "type", "const", "minimum", "maximum",
    "minLength", "maxLength", "pattern", "minProperties", "properties",
    "propertyNames", "required", "additionalProperties", "minItems",
    "maxItems", "uniqueItems", "items",
}


class SchemaGateError(ValueError):
    pass


def _json_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (set(left) == set(right)
                and all(_json_equal(left[key], right[key]) for key in left))
    if isinstance(left, list):
        return (len(left) == len(right)
                and all(_json_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _type_matches(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return type(value) is int
    if expected == "null":
        return value is None
    if expected == "boolean":
        return type(value) is bool
    raise SchemaGateError(f"unsupported schema type {expected!r}")


def _check_schema(schema, path):
    if not isinstance(schema, dict):
        raise SchemaGateError(f"{path}: schema must be an object")
    unsupported = set(schema) - _SUPPORTED
    if unsupported:
        raise SchemaGateError(
            f"{path}: unsupported schema keyword {sorted(unsupported)[0]!r}")
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not types or not all(isinstance(item, str) for item in types):
            raise SchemaGateError(f"{path}: invalid type declaration")
        for item in types:
            if item not in {"object", "array", "string", "integer", "null", "boolean"}:
                raise SchemaGateError(f"{path}: unsupported schema type {item!r}")
    for key in ("properties",):
        value = schema.get(key, {})
        if not isinstance(value, dict):
            raise SchemaGateError(f"{path}: {key} must be an object")
        for name, child in value.items():
            if not isinstance(name, str):
                raise SchemaGateError(f"{path}: property name must be a string")
            _check_schema(child, f"{path}.{name}")
    for key in ("additionalProperties", "propertyNames", "items"):
        child = schema.get(key)
        if child is not None and type(child) is not bool:
            _check_schema(child, f"{path}.{key}")


@functools.lru_cache(maxsize=1)
def _load_schema():
    try:
        with SCHEMA_PATH.open(encoding="ascii") as handle:
            schema = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaGateError(f"cannot load {SCHEMA_PATH.name}: {exc}") from exc
    _check_schema(schema, "$")
    return schema


def _validate(value, schema, path, errors):
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(value, item) for item in types):
            errors.append(f"{path}: expected {' or '.join(types)}")
            return

    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: value must equal {schema['const']!r}")

    if type(value) is int:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value exceeds maximum {schema['maximum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string is longer than {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: string does not match {schema['pattern']!r}")

    if isinstance(value, dict):
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(f"{path}: object has fewer than {schema['minProperties']} properties")
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", {})
        name_schema = schema.get("propertyNames")
        for name, item in value.items():
            if name_schema is not None:
                _validate(name, name_schema, f"{path}.{name}<property>", errors)
            if name in properties:
                _validate(item, properties[name], f"{path}.{name}", errors)
            elif additional is False:
                errors.append(f"{path}: additional property {name!r} is forbidden")
            elif isinstance(additional, dict):
                _validate(item, additional, f"{path}.{name}", errors)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: array has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: array has more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if any(_json_equal(item, prior) for prior in value[:index]):
                    errors.append(f"{path}[{index}]: duplicate array item")
                    break
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]", errors)


def instance_schema_errors(instance):
    errors = []
    _validate(instance, _load_schema(), "$", errors)
    return errors
