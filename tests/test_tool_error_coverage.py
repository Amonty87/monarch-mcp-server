"""Every failure a tool reports must reach MCP clients with ``isError: true``.

``test_tool_errors`` covers the mechanism: ``json_error`` and ``json_rejected``
return a ``ToolErrorText`` that the registration wrapper turns into an error
result. That left every failure path that built its own body -- local argument
validation, "no such id", Monarch-side rejections wrapped in ``json_success``,
and one hand-rolled ``json.dumps`` -- reporting failure in the body while the
protocol flag said the call succeeded. In edit mode that meant a refused rule
write looked like a successful one to any client that trusts the flag.

These tests go through a real MCP client session, because the flag is set by
the protocol layer and a direct function call cannot show it.
"""

import json
from unittest.mock import patch

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from monarch_mcp_server import tool_errors
from monarch_mcp_server.app import mcp
from monarch_mcp_server.helpers import json_success

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp < 2.0
    from mcp.server.fastmcp import FastMCP


async def _call(server, name, arguments):
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as client:
        return await client.call_tool(name, arguments)


def _text(result):
    return "".join(getattr(block, "text", "") for block in result.content)


def _rules(*rules):
    return {"transactionRules": list(rules)}


_MERCHANT_RULE = {
    "id": "r1",
    "order": 0,
    "merchantNameCriteria": [{"operator": "contains", "value": "x"}],
}
_REJECTED = {"message": "nope", "code": "INVALID"}


def _fail(exc):
    def setup(client):
        client.get_transactions.side_effect = exc

    return setup


def _gql(*responses):
    def setup(client):
        if len(responses) == 1:
            client.gql_call.return_value = responses[0]
        else:
            client.gql_call.side_effect = list(responses)

    return setup


def _no_client(client):
    """The failure is decided before any Monarch call."""


# (case id, tool, arguments, client setup, text the error body must carry)
FAILURE_PATHS = [
    # The hand-rolled json.dumps in get_transactions' except block.
    (
        "get_transactions/exception",
        "get_transactions",
        {"limit": 1, "start_date": "not-a-date", "end_date": "2026-09-30"},
        _fail(ValueError("bad date")),
        '"tool": "get_transactions"',
    ),
    # Local argument validation.
    (
        "get_net_worth_by_account_type/bad-timeframe",
        "get_net_worth_by_account_type",
        {"start_date": "2026-01-01", "timeframe": "week"},
        _no_client,
        "timeframe must be",
    ),
    (
        "set_budget_amount/both-ids",
        "set_budget_amount",
        {"amount": 0, "category_id": "c", "category_group_id": "g"},
        _no_client,
        "Cannot specify both",
    ),
    (
        "set_budget_amount/no-id",
        "set_budget_amount",
        {"amount": 0},
        _no_client,
        "Must specify either",
    ),
    (
        "reorder_transaction_rule/negative-order",
        "reorder_transaction_rule",
        {"rule_id": "r1", "new_order": -1},
        _no_client,
        "new_order must be 0 or greater",
    ),
    (
        "create_transaction_rule/no-criteria",
        "create_transaction_rule",
        {"set_category_id": "c1"},
        _no_client,
        "at least one matching criterion",
    ),
    (
        "update_category/bad-budget-variability",
        "update_category",
        {"category_id": "c1", "budget_variability": "sometimes"},
        _no_client,
        "Invalid budget_variability",
    ),
    (
        "update_category/unconfirmed-rollover-reset",
        "update_category",
        {"category_id": "c1", "rollover_start_month": "2026-01-01"},
        _no_client,
        "confirm_rollover_reset",
    ),
    (
        "update_category/bad-rollover-frequency",
        "update_category",
        {"category_id": "c1", "rollover_frequency": "hourly"},
        _no_client,
        "Invalid rollover_frequency",
    ),
    (
        "update_category/nothing-to-update",
        "update_category",
        {"category_id": "c1"},
        _no_client,
        "At least one field",
    ),
    (
        "update_merchant/nothing-to-update",
        "update_merchant",
        {"merchant_id": "m1"},
        _no_client,
        "At least one field",
    ),
    (
        "update_savings_goal/nothing-to-update",
        "update_savings_goal",
        {"goal_id": "g1"},
        _no_client,
        "Nothing to update",
    ),
    (
        "upload_account_balance_history/no-corrections",
        "upload_account_balance_history",
        {"account_id": "12345", "corrections": "{}"},
        _no_client,
        "No corrections provided",
    ),
    (
        "upload_account_balance_history/no-matching-dates",
        "upload_account_balance_history",
        {"account_id": "12345", "corrections": '{"2026-01-01": 500.0}'},
        _no_client,
        "No matching dates",
    ),
    # The id the caller named does not exist.
    (
        "reorder_transaction_rule/unknown-rule",
        "reorder_transaction_rule",
        {"rule_id": "r9", "new_order": 0},
        _gql(_rules()),
        "No transaction rule found",
    ),
    (
        "update_transaction_rule/unknown-rule",
        "update_transaction_rule",
        {"rule_id": "r9", "set_category_id": "c1"},
        _gql(_rules()),
        "No transaction rule found",
    ),
    (
        "update_transaction_rule/no-criteria-to-resend",
        "update_transaction_rule",
        {"rule_id": "r1", "set_category_id": "c1"},
        _gql(_rules({"id": "r1", "order": 0})),
        "no merchant, statement or amount criteria",
    ),
    (
        "update_category/dry-run-unknown-category",
        "update_category",
        {"category_id": "c9", "name": "x", "dry_run": True},
        _gql({"category": None}),
        "No category found",
    ),
    (
        "get_category_details/unknown-category",
        "get_category_details",
        {"category_id": "c9"},
        _gql({"category": None}),
        "No category found",
    ),
    (
        "get_merchant/unknown-merchant",
        "get_merchant",
        {"merchant_id": "m9"},
        _gql({"merchant": None}),
        "No merchant found",
    ),
    # Monarch refused the write inside an HTTP 200.
    (
        "create_transaction_rule/rejected",
        "create_transaction_rule",
        {"merchant_criteria_value": "x", "set_category_id": "c1"},
        _gql({"createTransactionRuleV2": {"errors": _REJECTED}}),
        "nope",
    ),
    (
        "update_transaction_rule/rejected",
        "update_transaction_rule",
        {"rule_id": "r1", "set_category_id": "c1"},
        _gql(
            _rules(_MERCHANT_RULE),
            {"updateTransactionRuleV2": {"errors": _REJECTED}},
        ),
        "nope",
    ),
    (
        "delete_transaction_rule/rejected",
        "delete_transaction_rule",
        {"rule_id": "r1"},
        _gql({"deleteTransactionRule": {"errors": _REJECTED}}),
        "nope",
    ),
    (
        "update_category/rejected",
        "update_category",
        {"category_id": "c1", "name": "x"},
        _gql({"updateCategory": {"errors": _REJECTED}}),
        "nope",
    ),
    (
        "update_merchant/rejected",
        "update_merchant",
        {"merchant_id": "m1", "name": "x"},
        _gql({"updateMerchant": {"errors": _REJECTED}}),
        "nope",
    ),
    (
        "review_recurring_stream/rejected",
        "review_recurring_stream",
        {"stream_id": "s1", "review_status": "approved"},
        _gql({"reviewRecurringStream": {"errors": _REJECTED}}),
        "nope",
    ),
]


@pytest.mark.parametrize(
    "tool, arguments, setup, expected",
    [case[1:] for case in FAILURE_PATHS],
    ids=[case[0] for case in FAILURE_PATHS],
)
async def test_failure_path_sets_is_error(
    mock_monarch_client, tool, arguments, setup, expected
):
    setup(mock_monarch_client)

    result = await _call(mcp, tool, arguments)

    body = _text(result)
    assert result.isError is True, body
    # The JSON body is kept for clients that parse it.
    assert expected in body


class TestMonarchRejectionsNameTheTool:
    """Rejections now go through json_rejected, like the other writers."""

    async def test_rejected_rule_write_names_the_tool(self, mock_monarch_client):
        mock_monarch_client.gql_call.return_value = {
            "deleteTransactionRule": {"errors": _REJECTED}
        }

        result = await _call(mcp, "delete_transaction_rule", {"rule_id": "r1"})

        body = _text(result)
        payload = json.loads(body[body.index("{"):])
        assert payload["success"] is False
        assert payload["tool"] == "delete_transaction_rule"
        assert payload["errors"] == _REJECTED


class TestAuthDiagnosticsReportFailures:
    """A keyring read that raises is a failed call, not a status report."""

    @pytest.mark.parametrize("tool", ["check_auth_status", "debug_session_loading"])
    async def test_exception_sets_is_error(self, tool):
        with patch(
            "monarch_mcp_server.tools.auth.secure_session.load_session",
            side_effect=RuntimeError("keyring backend unavailable"),
        ):
            result = await _call(mcp, tool, {})

        assert result.isError is True
        assert "keyring backend unavailable" in _text(result)

    @pytest.mark.parametrize("tool", ["check_auth_status", "debug_session_loading"])
    async def test_missing_session_is_a_status_not_an_error(self, tool):
        with patch(
            "monarch_mcp_server.tools.auth.secure_session.load_session",
            return_value=None,
        ):
            result = await _call(mcp, tool, {})

        assert result.isError is False
        assert "No Monarch session" in _text(result)


class TestSuccessesStaySuccesses:
    async def test_dry_run_of_a_real_category_is_not_an_error(
        self, mock_monarch_client
    ):
        mock_monarch_client.gql_call.return_value = {
            "category": {"id": "c1", "name": "Old", "icon": "x"}
        }

        result = await _call(
            mcp, "update_category", {"category_id": "c1", "name": "x", "dry_run": True}
        )

        assert result.isError is False
        assert '"dry_run": true' in _text(result)


def _server_with(body_by_name):
    server = FastMCP("tool-error-invariant")
    tool_errors.install(server)
    for name, body in body_by_name.items():

        def make(value):
            async def tool() -> str:
                return value

            return tool

        fn = make(body)
        fn.__name__ = name
        server.tool()(fn)
    return server


class TestFailureBodiesAreErrorsForEveryTool:
    """The wrapper, not each tool, guarantees the flag matches the body.

    A tool added later, or merged from upstream, that builds its failure body
    with json_success must still be reported as an error. Together with
    ``test_every_registered_tool_is_wrapped`` this covers every tool.
    """

    FAILURES = {
        "success_false": json_success({"success": False, "message": "no"}),
        "error_true": json_success({"error": True, "message": "no"}),
        "compact": json.dumps({"success": False}, separators=(",", ":")),
    }
    NOT_FAILURES = {
        "success_true": json_success({"success": True}),
        "nested_success_false": json_success({"data": [{"success": False}]}),
        "error_detail_string": json_success({"errors": [{"error": "x"}]}),
        "json_list": json_success([{"success": False}]),
        "plain_text": "✅ Session found (success: false is not JSON)",
        "broken_json": '{"success": false',
    }

    @pytest.mark.parametrize("name", sorted(FAILURES))
    async def test_failure_body_sets_is_error(self, name):
        server = _server_with(self.FAILURES)

        result = await _call(server, name, {})

        assert result.isError is True
        body = _text(result)
        assert json.loads(body[body.index("{"):]) == json.loads(self.FAILURES[name])

    @pytest.mark.parametrize("name", sorted(NOT_FAILURES))
    async def test_other_bodies_are_not_errors(self, name):
        server = _server_with(self.NOT_FAILURES)

        result = await _call(server, name, {})

        assert result.isError is False
        assert _text(result) == self.NOT_FAILURES[name]

    async def test_every_registered_tool_is_wrapped(self):
        tools = mcp._tool_manager.list_tools()

        assert tools, "no tools registered"
        unwrapped = [
            t.name for t in tools if not tool_errors.reports_errors(t.fn)
        ]
        assert unwrapped == []
