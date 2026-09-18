"""Validation and execution of human-approved, agent-generated analysis tools."""

from __future__ import annotations

import ast
import json
import math
from typing import Any

import numpy as np
import pandas as pd

MAX_CODE_CHARS = 20_000
MAX_OUTPUT_BYTES = 1_000_000
_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
)
_FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def validate_dynamic_tools(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise ValueError("dynamic_tools must be an array")
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Every dynamic tool must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.startswith("dynamic_") or not name.replace("_", "").isalnum():
            raise ValueError("Dynamic tool names must use the dynamic_<name> identifier form")
        if name in result:
            raise ValueError(f"Duplicate dynamic tool: {name}")
        required = item.get("required_parameters")
        if not isinstance(required, list) or not required or not all(isinstance(key, str) and key for key in required):
            raise ValueError(f"Dynamic tool {name} requires a non-empty required_parameters array")
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            raise ValueError(f"Dynamic tool {name} requires a description")
        _validate_source(item.get("local_code"), "run_local")
        _validate_source(item.get("server_code"), "run_server")
        result[name] = item
    return result


def dynamic_manifests(tools: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "version": int(spec.get("version", 1)),
            "description": spec["description"],
            "required_parameters": spec["required_parameters"],
            "local_output": spec.get("local_output", "human-approved aggregate JSON"),
            "server_operation": spec.get("server_operation", "human-approved aggregation code"),
            "dynamic": True,
        }
        for name, spec in tools.items()
    }


def execute_dynamic_local(
    spec: dict[str, Any], analysis: dict[str, Any], data: pd.DataFrame, min_cell_count: int
) -> dict[str, Any]:
    fields = analysis.get("fields")
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields):
        raise ValueError("Dynamic local execution requires a non-empty fields parameter")
    missing = sorted(set(fields) - set(data.columns))
    if missing:
        raise ValueError(f"Dynamic tool fields are unavailable after harmonization: {missing}")
    function = _load_function(spec["local_code"], "run_local")
    output = function(data[fields].copy(), dict(analysis), int(min_cell_count))
    return _validate_output(output, min_cell_count=min_cell_count, local=True)


def execute_dynamic_server(
    spec: dict[str, Any], analysis: dict[str, Any], outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    function = _load_function(spec["server_code"], "run_server")
    output = function(outputs, dict(analysis))
    output = _validate_output(output, min_cell_count=0, local=False)
    if output.get("schema_version") != "biobank.dynamic_server_output.v1":
        raise ValueError("Dynamic server output requires schema_version biobank.dynamic_server_output.v1")
    # Status is part of the runtime envelope rather than the authored algorithm.
    # A successful, validated return is executable by definition.
    output.setdefault("status", "ok")
    return output


def _validate_source(source: Any, function_name: str) -> None:
    if not isinstance(source, str) or not source.strip() or len(source) > MAX_CODE_CHARS:
        raise ValueError(f"Dynamic tool {function_name} source is missing or too large")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Dynamic tool {function_name} source is invalid: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != function_name:
        raise ValueError(f"Dynamic source must contain only def {function_name}(...)")
    expected = 3 if function_name == "run_local" else 2
    if len(functions[0].args.args) != expected:
        raise ValueError(f"{function_name} requires exactly {expected} positional arguments")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"Dynamic source contains forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Dynamic source uses forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Dynamic source may not access private or dunder attributes")


def _load_function(source: str, function_name: str) -> Any:
    _validate_source(source, function_name)
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "np": np,
        "pd": pd,
    }
    exec(compile(source, f"<approved-{function_name}>", "exec"), namespace, namespace)
    return namespace[function_name]


def _validate_output(value: Any, *, min_cell_count: int, local: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Dynamic tool output must be a JSON object")
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dynamic tool output is not finite JSON: {exc}") from exc
    if len(encoded.encode()) > MAX_OUTPUT_BYTES:
        raise ValueError("Dynamic tool output exceeds the aggregate payload limit")
    if local:
        if value.get("schema_version") != "biobank.dynamic_local_output.v1":
            raise ValueError("Dynamic local output requires schema_version biobank.dynamic_local_output.v1")
        _check_disclosure_counts(value, min_cell_count)
    return value


def _check_disclosure_counts(value: Any, min_cell_count: int) -> None:
    if isinstance(value, dict):
        count = value.get("n")
        if isinstance(count, int) and 0 < count < min_cell_count:
            raise ValueError("Dynamic local output contains an exact count below min_cell_count")
        for child in value.values():
            _check_disclosure_counts(child, min_cell_count)
    elif isinstance(value, list):
        if len(value) > 2048:
            raise ValueError("Dynamic output contains an oversized array")
        for child in value:
            _check_disclosure_counts(child, min_cell_count)
