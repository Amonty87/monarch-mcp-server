"""Tests for goals MCP tools."""

import json
from unittest.mock import AsyncMock, patch

from monarch_mcp_server.tools.goals import (
    get_goals, update_savings_goal, get_goal_contributions,
    set_goal_contribution,
)


def _goal(**overrides):
    goal = {
        "id": "sg_1",
        "name": "Rainy Day Fund",
        "type": "emergency_fund",
        "priority": 2,
        "progress": 0.24,
        "currentBalance": 1782.13,
        "targetAmount": 7500.0,
        "targetDate": "2026-12-31",
        "plannedMonthlyContribution": 100.0,
        "archivedAt": None,
        "completedAt": None,
    }
    goal.update(overrides)
    return goal


def _client(goals):
    c = AsyncMock()
    c.gql_call.return_value = {"savingsGoals": goals}
    return c


class TestGetGoals:
    """Tests for get_goals tool."""

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_reads_savings_goals_not_goalsv2(self, mock_get_client):
        """The tool must query savingsGoals.

        goalsV2 is the superseded collection and serves pre-migration numbers.
        """
        client = _client([_goal()])
        mock_get_client.return_value = client

        data = json.loads(await get_goals())

        assert data["count"] == 1
        assert data["goals"][0]["target_amount"] == 7500.0
        assert data["goals"][0]["target_date"] == "2026-12-31"
        assert data["goals"][0]["current_balance"] == 1782.13
        # progress arrives as a 0..1 fraction and is surfaced as a percentage
        assert data["goals"][0]["progress_percent"] == 24.0
        # The operation name is the accessible signal that the right
        # collection was queried; the query itself is a parsed DocumentNode.
        assert client.gql_call.call_args.kwargs["operation"] == "GetSavingsGoals"

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_goalsv2_payload_yields_nothing(self, mock_get_client):
        """Guards against silently reading the wrong collection."""
        c = AsyncMock()
        c.gql_call.return_value = {"goalsV2": [_goal()]}
        mock_get_client.return_value = c

        assert json.loads(await get_goals())["count"] == 0

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_archived_at_is_raw_and_never_filtered(self, mock_get_client):
        """archivedAt is passed through; nothing is hidden from the caller."""
        mock_get_client.return_value = _client([
            _goal(), _goal(id="sg_2", archivedAt="2026-01-01T00:00:00+00:00"),
        ])

        data = json.loads(await get_goals())

        assert data["count"] == 2
        assert "archived" not in data["goals"][0]

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_error_handling(self, mock_get_client):
        """Transport failures are reported as tool errors."""
        c = AsyncMock()
        c.gql_call.side_effect = Exception("boom")
        mock_get_client.return_value = c

        data = json.loads(await get_goals())

        assert data["error"] is True
        assert data["tool"] == "get_goals"


def _update_client(errors=None, goal=None):
    c = AsyncMock()
    c.gql_call.return_value = {
        "updateSavingsGoal": {
            "savingsGoal": goal or {
                "id": "sg_1", "name": "Rainy Day Fund", "targetAmount": 7500.0,
                "targetDate": "2026-12-31", "plannedMonthlyContribution": 250.0,
                "priority": 2,
            },
            "errors": errors,
        }
    }
    return c


class TestUpdateSavingsGoal:
    """Tests for update_savings_goal tool."""

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_updates_target(self, mock_get_client):
        client = _update_client()
        mock_get_client.return_value = client

        data = json.loads(await update_savings_goal("sg_1", target_amount=8000.0))

        assert data["success"] is True
        sent = client.gql_call.call_args.kwargs["variables"]["input"]
        assert sent == {"id": "sg_1", "targetAmount": 8000.0}

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_only_supplied_fields_are_sent(self, mock_get_client):
        """Omitted fields are preserved by Monarch, so they must not be sent.

        Sending an unset field as null would clear it.
        """
        client = _update_client()
        mock_get_client.return_value = client

        await update_savings_goal("sg_1", target_amount=8000.0)

        sent = client.gql_call.call_args.kwargs["variables"]["input"]
        assert sent == {"id": "sg_1", "targetAmount": 8000.0}
        assert "targetDate" not in sent
        assert "name" not in sent

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_no_fields_is_rejected_without_a_call(self, mock_get_client):
        client = _update_client()
        mock_get_client.return_value = client

        data = json.loads(await update_savings_goal("sg_1"))

        assert data["success"] is False
        client.gql_call.assert_not_called()

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_reports_errors(self, mock_get_client):
        mock_get_client.return_value = _update_client(
            errors={"message": "bad", "code": None, "fieldErrors": None})

        data = json.loads(await update_savings_goal("sg_1", target_amount=1.0))

        assert data["success"] is False
        assert data["errors"]["message"] == "bad"

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_blank_error_payload_is_readable(self, mock_get_client):
        """An all-null PayloadError becomes a real message, not an empty dict."""
        mock_get_client.return_value = _update_client(
            errors={"message": None, "code": None, "fieldErrors": None})

        data = json.loads(await update_savings_goal("sg_1", target_amount=1.0))

        assert data["success"] is False
        assert data["errors"]["message"]

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_error_handling(self, mock_get_client):
        c = AsyncMock()
        c.gql_call.side_effect = Exception("boom")
        mock_get_client.return_value = c

        data = json.loads(await update_savings_goal("sg_1", target_amount=1.0))

        assert data["error"] is True
        assert data["tool"] == "update_savings_goal"

    async def test_contribution_is_not_settable(self):
        """The monthly contribution must not be exposed.

        Monarch accepts plannedMonthlyContribution on this input and reports
        success, but the value does not persist -- verified against the live
        API. Accepting the argument would report a change that never happened.
        """
        import inspect
        from monarch_mcp_server.tools.goals import update_savings_goal as fn
        params = inspect.signature(fn).parameters
        assert "planned_monthly_contribution" not in params

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_type_and_sinking_fund_are_settable(self, mock_get_client):
        """Both persist on the live API, so both are exposed."""
        client = _update_client()
        mock_get_client.return_value = client

        await update_savings_goal("sg_1", goal_type="sinking_fund",
                                  is_sinking_fund=True)

        sent = client.gql_call.call_args.kwargs["variables"]["input"]
        assert sent == {"id": "sg_1", "type": "sinking_fund",
                        "isSinkingFund": True}

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_false_is_honoured_not_treated_as_unset(self, mock_get_client):
        """is_sinking_fund=False must be sent, not dropped as falsy."""
        client = _update_client()
        mock_get_client.return_value = client

        await update_savings_goal("sg_1", is_sinking_fund=False)

        sent = client.gql_call.call_args.kwargs["variables"]["input"]
        assert sent == {"id": "sg_1", "isSinkingFund": False}


class TestGoalContributions:
    """Tests for goal contribution tools."""

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_get_contributions_breaks_down_by_account(self, mock_get_client):
        c = AsyncMock()
        c.gql_call.return_value = {"savingsGoal": {
            "id": "sg_1", "name": "Rainy Day Fund",
            "monthlyBudgetAmounts": [{
                "month": "2026-08-01", "totalPlannedAmount": 1143.0,
                "totalActualAmount": 0.0, "totalRemainingAmount": 1143.0,
                "accountBreakdown": [
                    {"account": {"id": "a1", "displayName": "Savings"},
                     "plannedAmount": 143.0, "actualAmount": 0.0,
                     "remainingAmount": 143.0},
                ],
            }],
        }}
        mock_get_client.return_value = c

        data = json.loads(await get_goal_contributions("sg_1", month="2026-08"))

        assert data["months"][0]["accounts"][0]["planned"] == 143.0
        sent = c.gql_call.call_args.kwargs["variables"]
        assert sent["startMonth"] == "2026-08-01"
        assert sent["endMonth"] == "2026-08-31"

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_set_contribution_sends_only_that_account(self, mock_get_client):
        """Unmentioned accounts are preserved by Monarch, so only one is sent.

        Resending the whole allocation risks duplicate-account errors and would
        overwrite concurrent edits to other accounts.
        """
        c = AsyncMock()
        c.gql_call.return_value = {"updateSavingsGoal": {
            "savingsGoal": {"id": "sg_1"}, "errors": None}}
        mock_get_client.return_value = c

        data = json.loads(await set_goal_contribution("sg_1", "a1", 175.0))

        assert data["success"] is True
        sent = c.gql_call.call_args.kwargs["variables"]["input"]
        assert sent == {"id": "sg_1",
                        "accountBudgetAmounts": [{"accountId": "a1", "amount": 175.0}]}

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_set_contribution_zero_is_sent(self, mock_get_client):
        """0 removes the contribution and must not be dropped as falsy."""
        c = AsyncMock()
        c.gql_call.return_value = {"updateSavingsGoal": {
            "savingsGoal": {"id": "sg_1"}, "errors": None}}
        mock_get_client.return_value = c

        await set_goal_contribution("sg_1", "a1", 0.0)

        sent = c.gql_call.call_args.kwargs["variables"]["input"]
        assert sent["accountBudgetAmounts"][0]["amount"] == 0.0

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_set_contribution_reports_errors(self, mock_get_client):
        c = AsyncMock()
        c.gql_call.return_value = {"updateSavingsGoal": {
            "savingsGoal": None,
            "errors": {"message": "Duplicate account_id", "fieldErrors": None}}}
        mock_get_client.return_value = c

        data = json.loads(await set_goal_contribution("sg_1", "a1", 1.0))

        assert data["success"] is False
        assert "Duplicate" in data["errors"]["message"]


class TestGetGoalsDocstring:
    """The docstring is the tool description a model reads, so every output
    field it names has to exist. It promised default_name and objective after
    the switch to savingsGoals dropped both (objective is confirmed absent on
    SavingsGoal), so a model matching goals on them found nothing."""

    # Backticked names in the docstring that are not get_goals output fields.
    NOT_OUTPUT_FIELDS = {"link_goal_id"}

    @patch('monarch_mcp_server.tools.goals.get_monarch_client')
    async def test_documented_fields_are_returned(self, mock_get_client):
        import re

        mock_get_client.return_value = _client([_goal()])
        returned = set(json.loads(await get_goals())["goals"][0])

        documented = set(re.findall(r"`([a-z][a-z0-9_]*)`", get_goals.__doc__))
        missing = documented - self.NOT_OUTPUT_FIELDS - returned

        assert not missing, f"docstring names fields get_goals never returns: {missing}"
