"""Failed tool calls must reach MCP clients as errors.

Every tool catches its own exceptions and returns a JSON body from
``json_error`` or ``json_rejected``. Returning, rather than raising, meant the
MCP result carried ``isError: false``, so a client that trusts the protocol
flag (Claude Desktop, and any model reading it) saw a failed or refused write
as a successful call.

These tests go through a real MCP client session, because the flag is set by
the protocol layer and a direct function call cannot show it.
"""

import json

from mcp.shared.memory import create_connected_server_and_client_session

from monarch_mcp_server.app import mcp
from monarch_mcp_server.helpers import json_error, json_rejected

REJECTION = {
    "updateTransaction": {
        "transaction": None,
        "errors": {"message": "Category not found", "code": "INVALID"},
    }
}


async def _call(name, arguments):
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        return await client.call_tool(name, arguments)


def _text(result):
    return "".join(getattr(block, "text", "") for block in result.content)


class TestFailuresAreReportedAsErrors:
    async def test_exception_inside_a_tool_sets_is_error(self, mock_monarch_client):
        mock_monarch_client.get_transaction_details.side_effect = Exception("Not found")

        result = await _call("get_transaction_details", {"transaction_id": "0"})

        assert result.isError is True
        body = _text(result)
        assert "Not found" in body
        # The JSON body is kept so clients that parse it still can.
        assert '"error": true' in body
        assert '"tool": "get_transaction_details"' in body

    async def test_payload_rejection_sets_is_error(self, mock_monarch_client):
        mock_monarch_client.update_transaction.return_value = REJECTION

        result = await _call(
            "update_transaction_notes", {"transaction_id": "t", "notes": "n"}
        )

        assert result.isError is True
        body = _text(result)
        assert "Category not found" in body
        assert '"success": false' in body

    async def test_success_is_not_an_error(self, mock_monarch_client):
        result = await _call(
            "update_transaction_notes", {"transaction_id": "t", "notes": "n"}
        )

        assert result.isError is False
        assert "updateTransaction" in _text(result)


class TestDirectCallersAreUnchanged:
    """server.py re-exports the tool functions, and the suite calls them
    directly. Those callers keep getting the JSON text they always did."""

    def test_json_error_is_still_json_text(self):
        body = json_error("some_tool", RuntimeError("boom"))
        assert isinstance(body, str)
        assert json.loads(body) == {
            "error": True,
            "tool": "some_tool",
            "message": "boom",
        }

    def test_json_rejected_is_still_json_text(self):
        body = json_rejected("some_tool", {"message": "nope"})
        assert isinstance(body, str)
        assert json.loads(body)["success"] is False

    async def test_direct_call_returns_text_not_raise(self, mock_monarch_client):
        from monarch_mcp_server.tools.transactions import get_transaction_details

        mock_monarch_client.get_transaction_details.side_effect = Exception("x")
        body = await get_transaction_details("0")
        assert json.loads(body)["error"] is True


class TestRegistrationIsUnchanged:
    async def test_input_schema_survives_the_wrapper(self):
        """The wrapper must not hide a tool's parameters from FastMCP.

        goals.py uses postponed annotations, so it is the case most likely to
        break if the wrapper's signature were resolved in the wrong module.
        """
        tools = {t.name: t for t in await mcp.list_tools()}
        schema = tools["get_goal_contributions"].inputSchema
        assert "goal_id" in schema["properties"]
        assert schema["required"] == ["goal_id"]
        notes = tools["update_transaction_notes"].inputSchema
        assert set(notes["required"]) == {"transaction_id", "notes"}
