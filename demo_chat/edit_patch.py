"""Aider-style source edits for a single, self-contained Studio page.

The model returns literal replacements. The caller supplies the source snapshot
from the request and decides when the completed response may be committed.
"""

import json
import re


SEARCH = "<<<<<<< SEARCH"
SEPARATOR = "======="
REPLACE = ">>>>>>> REPLACE"
MARKERS = frozenset((SEARCH, SEPARATOR, REPLACE))
MAX_EDITS = 4

PATCH_SYSTEM_PROMPT = f"""Edit a self-contained HTML page by returning only SEARCH/REPLACE blocks. Preserve all unrelated code and behavior. Keep CSS and JavaScript inline, keep the layout responsive, and use no external dependencies or network calls.

Use the following format, with each marker on its own line:
{SEARCH}
exact existing code
{SEPARATOR}
new code
{REPLACE}

Copy the smallest existing fragment that occurs exactly once, including its whitespace and capitalization. Search fragments are literal substrings of the original page. You can use up to four disjoint blocks. To delete a fragment, leave the replacement empty. Do not add line numbers, a file name, commentary, or the full HTML document. If the change requires a full rewrite, respond with REWRITE."""

JSON_PATCH_SYSTEM_PROMPT = (
    'Edit a self-contained HTML page. Return only a JSON object with keys "search" and "replace", both strings. '
    'The search string must be a small exact substring that occurs once in the original HTML. '
    'The replace string contains just the updated code. Preserve all unrelated code. '
    'For example, {"search":"count += 1;","replace":"count += 3;"} changes the increment by replacing one statement. '
    'For behavior changes, find and edit the existing JavaScript statement. Avoid new scripts or duplicate handlers. '
    'If a small edit cannot implement the request, return {"rewrite":true}.'
)


def build_patch_prompt(source, instruction):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Provide the current HTML before requesting a patch.")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Describe the change to make.")
    fence_length = max((len(run) for run in re.findall(r"`+", source)), default=0) + 1
    fence = "`" * max(3, fence_length)
    return f"Current HTML:\n{fence}html\n{source}\n{fence}\n\nRequested edit:\n{instruction.strip()}"


def _parse_blocks(response):
    if not isinstance(response, str):
        raise ValueError("The edit response must be text.")
    response = response.replace("\r\n", "\n").strip()
    if response == "REWRITE":
        raise ValueError("Qwen requested a full rewrite.")
    lines = response.split("\n")
    if lines[0] in ("```", "```text", "```diff", "```patch"):
        if len(lines) < 3 or lines[-1] != "```":
            raise ValueError("The edit response contains an incomplete code fence.")
        lines = lines[1:-1]

    edits = []
    index = 0
    while index < len(lines):
        if lines[index] == "":
            index += 1
            continue
        if lines[index] != SEARCH or len(edits) == MAX_EDITS:
            raise ValueError("Expected up to four SEARCH/REPLACE blocks with no surrounding text.")
        index += 1
        original = []
        while index < len(lines) and lines[index] not in MARKERS:
            original.append(lines[index])
            index += 1
        if index == len(lines) or lines[index] != SEPARATOR:
            raise ValueError("The SEARCH block is missing its separator.")
        index += 1
        replacement = []
        while index < len(lines) and lines[index] not in MARKERS:
            replacement.append(lines[index])
            index += 1
        if index == len(lines) or lines[index] != REPLACE:
            raise ValueError("The replacement block is incomplete.")
        edits.append(("\n".join(original), "\n".join(replacement)))
        index += 1
    if not edits:
        raise ValueError("The edit response contains no replacements.")
    return edits


def apply_model_patch(source, response):
    """Return (updated HTML, edit count) after verifying every match.

    Resolve all ranges in the original snapshot. One edit cannot create a
    match for another, and a malformed or ambiguous edit rejects the batch.
    """
    return _apply_replacements(source, _parse_blocks(response), convert_line_endings=True)


def _unique_fields(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("The edit response contains duplicate JSON fields.")
    return fields


def apply_json_model_patch(source, response):
    """Apply one exact replacement from a completed JSON response."""
    if not isinstance(response, str):
        raise ValueError("The edit response must be text.")
    response = response.replace("\r\n", "\n").strip()
    first, _, body = response.partition("\n")
    if first in ("```", "```json"):
        if not body.endswith("\n```"):
            raise ValueError("The edit response contains an incomplete code fence.")
        response = body.removesuffix("\n```")
    try:
        fields = json.loads(response, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as error:
        raise ValueError("The edit response is not valid JSON.") from error
    if fields == {"rewrite": True}:
        raise ValueError("Qwen requested a full rewrite.")
    if (not isinstance(fields, dict) or set(fields) != {"search", "replace"}
            or not all(isinstance(value, str) for value in fields.values())):
        raise ValueError("Expected one literal search and replacement in the JSON edit response.")
    return _apply_replacements(source, [(fields["search"], fields["replace"])])


def _apply_replacements(source, edits, *, convert_line_endings=False):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("The edit needs its original HTML snapshot.")
    uses_crlf = convert_line_endings and "\r\n" in source and "\n" not in source.replace("\r\n", "")
    ranges = []
    for original, replacement in edits:
        if not original.strip():
            raise ValueError("A SEARCH fragment must contain code.")
        if uses_crlf:
            original = original.replace("\n", "\r\n")
            replacement = replacement.replace("\n", "\r\n")
        start = source.find(original)
        if start == -1:
            raise ValueError("A SEARCH fragment does not exactly match the current HTML.")
        if source.find(original, start + 1) != -1:
            raise ValueError("A SEARCH fragment matches more than one place in the current HTML.")
        ranges.append((start, start + len(original), replacement))

    pieces = []
    position = 0
    for start, end, replacement in sorted(ranges):
        if start < position:
            raise ValueError("SEARCH fragments overlap in the current HTML.")
        pieces.extend((source[position:start], replacement))
        position = end
    pieces.append(source[position:])
    return "".join(pieces), len(ranges)
