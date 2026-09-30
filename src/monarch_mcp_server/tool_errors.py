"""Report failed tool calls to MCP clients as errors.

Every tool catches its own exceptions and returns a JSON body built by
``helpers.json_error`` or ``helpers.json_rejected``. FastMCP only sets
``isError: true`` when a tool raises, so those failures reached clients as
successful calls unless the client parsed the body.

Like ``read_only``, this works by wrapping ``mcp.tool()`` at registration: the
function FastMCP registers raises when the tool returns a ``ToolErrorText``,
while the decorated function handed back to the tool module is the original,
so direct callers (``server.py`` re-exports, the test suite) still get the same
JSON text. The raised message is that JSON body, so clients that parse it keep
working.
"""

import functools
import inspect
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


def _raise_if_error(result: Any) -> Any:
    if isinstance(result, ToolErrorText):
        raise ToolError(str(result))
    return result


def _error_signalling(fn: F) -> F:
    """Wrap *fn* so a ``ToolErrorText`` result becomes a raised ToolError."""
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
