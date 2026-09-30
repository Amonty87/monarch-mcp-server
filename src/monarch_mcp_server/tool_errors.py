"""Report failed tool calls to MCP clients as errors.

Every tool catches its own exceptions and returns a JSON body built by
``helpers.json_error``, ``helpers.json_rejected`` or ``helpers.json_failure``.
FastMCP only sets ``isError: true`` when a tool raises, so those failures
reached clients as successful calls unless the client parsed the body.

Like ``read_only``, this works by wrapping ``mcp.tool()`` at registration: the
function FastMCP registers raises when the tool returns a failure, while the
decorated function handed back to the tool module is the original, so direct
callers (``server.py`` re-exports, the test suite) still get the same JSON text.

A result counts as a failure when it is a ``ToolErrorText``, or when it is a
JSON object whose top level carries ``"error": true`` or ``"success": false``.
The second rule is the backstop: a tool that builds its failure body with
``json_success`` (an upstream merge, a tool added later) is still reported as an
error, so the flag cannot disagree with the body.

FastMCP prefixes the text of an error result with ``Error executing tool
<name>: ``, so a client that parses the JSON body has to start at the first
``{``. The body itself is unchanged.
"""

import functools
import inspect
import json
import re
from typing import Any, Callable, TypeVar

from monarch_mcp_server.helpers import ToolErrorText

try:  # mcp >= 2.0 renamed the FastMCP package to mcpserver
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    try:
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError:  # any exception sets isError; keep the server starting

        class ToolError(Exception):  # type: ignore[no-redef]
            """Fallback when neither mcp layout provides ToolError."""


F = TypeVar("F", bound=Callable[..., Any])

# Cheap screen so a large successful body is only parsed when it could be a
# failure. json.dumps writes these with or without a space after the colon.
_FAILURE_MARKER = re.compile(r'"(?:success|error)"\s*:\s*(?:false|true)')

_MARKER_ATTR = "__monarch_reports_errors__"


def is_failure_body(result: Any) -> bool:
    """Whether *result* is a JSON object whose top level reports a failure."""
    if not isinstance(result, str) or not _FAILURE_MARKER.search(result):
        return False
    try:
        body = json.loads(result)
    except ValueError:
        return False
    return isinstance(body, dict) and (
        body.get("error") is True or body.get("success") is False
    )


def _raise_if_error(result: Any) -> Any:
    if isinstance(result, ToolErrorText) or is_failure_body(result):
        raise ToolError(str(result))
    return result


def reports_errors(fn: Any) -> bool:
    """Whether *fn* is a registered tool callable wrapped by this module."""
    return getattr(fn, _MARKER_ATTR, False) is True


def _error_signalling(fn: F) -> F:
    """Wrap *fn* so a failure result becomes a raised ToolError."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _raise_if_error(await fn(*args, **kwargs))

    else:

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _raise_if_error(fn(*args, **kwargs))

    # FastMCP resolves string annotations against the callable's __globals__,
    # which for this wrapper would be this module. Hand it a signature already
    # resolved in the tool's own module (goals.py uses postponed annotations).
    wrapper.__signature__ = inspect.signature(fn, eval_str=True)  # type: ignore[attr-defined]
    setattr(wrapper, _MARKER_ATTR, True)
    return wrapper  # type: ignore[return-value]


def install(mcp: Any) -> None:
    """Make every tool registered after this call report failures as errors."""
    original_tool = mcp.tool

    def error_signalling_tool(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        register = original_tool(*args, **kwargs)

        def decorator(fn: F) -> F:
            register(_error_signalling(fn))
            return fn

        return decorator

    mcp.tool = error_signalling_tool  # type: ignore[method-assign]
