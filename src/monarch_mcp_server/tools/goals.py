"""Goals tools with GraphQL queries."""

from __future__ import annotations

import calendar
import logging
from datetime import date
from typing import Any, Dict, Optional, Tuple

from gql import gql

from monarch_mcp_server.app import mcp
from monarch_mcp_server.client import get_monarch_client
from monarch_mcp_server.helpers import (
    json_error,
    json_rejected,
    json_success,
    payload_errors,
)

logger = logging.getLogger(__name__)

# Monarch disables GraphQL introspection for non-admin users and masks unknown
# field errors as a generic "Something went wrong", so this selection set was
# established field-by-field against the live API, then reconciled against the
# web app's own GoalSummaryFields fragment. Note the balance field is
# `currentBalance`, not `currentAmount`, and progress is served directly as a
# 0..1 fraction rather than being computed client-side.
#
# There are TWO goal collections. `savingsGoals` is the current one and matches
# what the app shows. `goalsV2` is the superseded collection: on a real account
# every goalsV2 record carried an identical archivedAt to the microsecond (the
# migration timestamp) while the same goals showed as active in the app, and
# their targets were stale -- a goal reading 10000 in goalsV2 was 7500 in
# savingsGoals. Query savingsGoals; goalsV2 will quietly serve pre-migration
# numbers.
#
# The two also have separate ids for the same goal, which is why a transaction
# rule has both linkGoalAction and linkSavingsGoalAction and they are NOT
# interchangeable.
GET_GOALS_QUERY = gql("""
query GetSavingsGoals {
  savingsGoals {
    id
    name
    type
    status
    priority
    progress
    currentBalance
    targetAmount
    targetDate
    plannedMonthlyContribution
    estimatedMonthsUntilCompletion
    forecastedCompletionDate
    isSinkingFund
    createdAt
    archivedAt
    completedAt
    __typename
  }
}
""")


@mcp.tool()
async def get_goals() -> str:
    """
    List Monarch savings and debt-paydown goals.

    Use this to find a goal id for `link_goal_id` on a transaction rule, or to
    report on goal progress.

    `name` is user-editable, so match goals programmatically on `id`.

    Every goal is returned. `archived_at` is passed through raw and should not
    be read as the app's archive state -- see the comment above the query.

    Returns:
        JSON list of goals with progress and the accounts allocated to each.
    """
    try:
        client = await get_monarch_client()
        result = await client.gql_call(
            operation="GetSavingsGoals", graphql_query=GET_GOALS_QUERY, variables={}
        )

        goals = []
        for g in result.get("savingsGoals") or []:
            goals.append({
                "id": g.get("id"),
                "name": g.get("name"),
                "type": g.get("type"),
                "status": g.get("status"),
                "priority": g.get("priority"),
                "current_balance": g.get("currentBalance"),
                "progress_percent": (
                    round(g["progress"] * 100, 1)
                    if isinstance(g.get("progress"), (int, float)) else None
                ),
                "target_amount": g.get("targetAmount"),
                "target_date": g.get("targetDate"),
                "planned_monthly_contribution": g.get("plannedMonthlyContribution"),
                "estimated_months_until_completion": g.get(
                    "estimatedMonthsUntilCompletion"),
                "forecasted_completion_date": g.get("forecastedCompletionDate"),
                "is_sinking_fund": g.get("isSinkingFund"),
                "created_at": g.get("createdAt"),
                "archived_at": g.get("archivedAt"),
                "completed_at": g.get("completedAt"),
            })

        return json_success({
            "count": len(goals),
            "note": (
                "Goals track allocated ACCOUNT BALANCES, not categorized "
                "transactions. A transaction rule only touches a goal via "
                "link_goal_id, which Monarch requires account_ids alongside. "
                "`archived_at` is raw and does not match the app's archive "
                "state -- do not filter on it."
            ),
            "goals": goals,
        })
    except Exception as e:
        return json_error("get_goals", e)


UPDATE_SAVINGS_GOAL_MUTATION = gql("""
mutation Common_UpdateSavingsGoal($input: UpdateSavingsGoalInput!) {
  updateSavingsGoal(input: $input) {
    savingsGoal {
      id
      name
      targetAmount
      targetDate
      plannedMonthlyContribution
      priority
      __typename
    }
    errors {
      message
      code
      fieldErrors {
        field
        messages
        __typename
      }
      __typename
    }
    __typename
  }
}
""")


@mcp.tool()
async def update_savings_goal(
    goal_id: str,
    target_amount: Optional[float] = None,
    target_date: Optional[str] = None,
    name: Optional[str] = None,
    priority: Optional[int] = None,
    goal_type: Optional[str] = None,
    is_sinking_fund: Optional[bool] = None,
) -> str:
    """
    Update a savings goal's target or monthly contribution.

    Unlike the transaction-rule mutation, this is a genuine partial update:
    omitted fields are preserved, verified against the live API. Pass only what
    you want to change.

    NOTE: the monthly contribution is not set here. `plannedMonthlyContribution`
    is a read-only rollup -- the input accepts it and reports success, but the
    value never persists. Contributions are budgeted per funding account; use
    `set_goal_contribution`.

    Args:
        goal_id: Goal to update. Use get_goals -- and note these are
            `savingsGoals` ids, which differ from the legacy goalsV2 ids.
        target_amount: Total amount the goal is aiming at.
        target_date: Target completion date, "YYYY-MM-DD". Optional on a goal.
        name: Rename the goal.
        priority: Ordering among goals, lower first.
        goal_type: Goal category, e.g. "emergency_fund", "retirement",
            "sinking_fund". Changes how Monarch treats the goal, so it is not
            merely a label.
        is_sinking_fund: Whether the goal is spent down and refilled rather
            than accumulated.

    Not exposed: the image fields (`imageStorageProvider` /
    `imageStorageProviderId`) are settable but purely cosmetic. Everything else
    on the input either is not accepted (`archivedAt`, `completedAt`, `icon`,
    `color`, `accountIds`) or does not persist (see the note above).

    Returns:
        JSON with the goal's state after the update.
    """
    try:
        changes: Dict[str, Any] = {}
        if target_amount is not None:
            changes["targetAmount"] = target_amount
        if target_date is not None:
            changes["targetDate"] = target_date
        if name is not None:
            changes["name"] = name
        if priority is not None:
            changes["priority"] = priority
        if goal_type is not None:
            changes["type"] = goal_type
        if is_sinking_fund is not None:
            changes["isSinkingFund"] = is_sinking_fund

        if not changes:
            return json_success({
                "success": False,
                "message": "Nothing to update -- pass at least one field.",
            })

        client = await get_monarch_client()
        result = await client.gql_call(
            operation="Common_UpdateSavingsGoal",
            graphql_query=UPDATE_SAVINGS_GOAL_MUTATION,
            variables={"input": {"id": goal_id, **changes}},
        )

        errors = payload_errors(result, "updateSavingsGoal")
        if errors:
            return json_rejected("update_savings_goal", errors)

        payload = result.get("updateSavingsGoal") or {}

        goal = payload.get("savingsGoal") or {}
        return json_success({
            "success": True,
            "goal_id": goal_id,
            "changed": sorted(changes),
            "goal": {
                "id": goal.get("id"),
                "name": goal.get("name"),
                "target_amount": goal.get("targetAmount"),
                "target_date": goal.get("targetDate"),
                "planned_monthly_contribution": goal.get(
                    "plannedMonthlyContribution"),
                "priority": goal.get("priority"),
            },
        })
    except Exception as e:
        return json_error("update_savings_goal", e)


GOAL_CONTRIBUTIONS_QUERY = gql("""
query GetSavingsGoalContributions($id: ID!, $startMonth: Date!, $endMonth: Date!) {
  savingsGoal(id: $id) {
    id
    name
    monthlyBudgetAmounts(startMonth: $startMonth, endMonth: $endMonth) {
      month
      totalPlannedAmount
      totalActualAmount
      totalRemainingAmount
      accountBreakdown {
        account {
          id
          displayName
          __typename
        }
        plannedAmount
        actualAmount
        remainingAmount
        __typename
      }
      __typename
    }
    __typename
  }
}
""")


def _month_bounds(month: Optional[str]) -> Tuple[str, str]:
    """(first, last) day of *month* ("YYYY-MM-DD" or "YYYY-MM"), default current."""
    if month:
        parts = month.split("-")
        year, mon = int(parts[0]), int(parts[1])
    else:
        today = date.today()
        year, mon = today.year, today.month
    last = calendar.monthrange(year, mon)[1]
    return f"{year:04d}-{mon:02d}-01", f"{year:04d}-{mon:02d}-{last:02d}"


@mcp.tool()
async def get_goal_contributions(goal_id: str, month: Optional[str] = None) -> str:
    """
    Show a goal's budgeted contributions, broken down by funding account.

    A goal's contribution is not one number -- it is budgeted per account, and
    the goal-level `plannedMonthlyContribution` is a rollup. Use this to see the
    per-account amounts and to get the `account_id` values that
    `set_goal_contribution` needs.

    Args:
        goal_id: A `savingsGoals` id (see get_goals).
        month: "YYYY-MM" or "YYYY-MM-DD". Defaults to the current month.

    Returns:
        JSON with the month's planned/actual totals and the per-account split.
    """
    try:
        start, end = _month_bounds(month)
        client = await get_monarch_client()
        result = await client.gql_call(
            operation="GetSavingsGoalContributions",
            graphql_query=GOAL_CONTRIBUTIONS_QUERY,
            variables={"id": goal_id, "startMonth": start, "endMonth": end},
        )
        goal = result.get("savingsGoal") or {}
        months = []
        for m in goal.get("monthlyBudgetAmounts") or []:
            months.append({
                "month": m.get("month"),
                "total_planned": m.get("totalPlannedAmount"),
                "total_actual": m.get("totalActualAmount"),
                "total_remaining": m.get("totalRemainingAmount"),
                "accounts": [
                    {
                        "account_id": (a.get("account") or {}).get("id"),
                        "name": (a.get("account") or {}).get("displayName"),
                        "planned": a.get("plannedAmount"),
                        "actual": a.get("actualAmount"),
                        "remaining": a.get("remainingAmount"),
                    }
                    for a in (m.get("accountBreakdown") or [])
                ],
            })
        return json_success({
            "goal_id": goal.get("id"),
            "name": goal.get("name"),
            "months": months,
        })
    except Exception as e:
        return json_error("get_goal_contributions", e)


@mcp.tool()
async def set_goal_contribution(
    goal_id: str, account_id: str, amount: float
) -> str:
    """
    Set the budgeted monthly contribution to a goal from one funding account.

    Contributions are budgeted per account, not per goal, which is why the
    goal-level `plannedMonthlyContribution` cannot be written to directly.

    Accounts you do not mention are left alone -- verified against the live API,
    so there is no need to resend the whole allocation. Use
    `get_goal_contributions` to find `account_id` values.

    Args:
        goal_id: A `savingsGoals` id (see get_goals).
        account_id: The funding account to budget from.
        amount: Monthly amount. 0 removes this account's contribution.
    """
    try:
        client = await get_monarch_client()
        result = await client.gql_call(
            operation="Common_UpdateSavingsGoal",
            graphql_query=UPDATE_SAVINGS_GOAL_MUTATION,
            variables={"input": {
                "id": goal_id,
                "accountBudgetAmounts": [
                    {"accountId": account_id, "amount": amount}
                ],
            }},
        )
        errors = payload_errors(result, "updateSavingsGoal")
        if errors:
            return json_rejected("set_goal_contribution", errors)

        payload = result.get("updateSavingsGoal") or {}
        return json_success({
            "success": True,
            "goal_id": goal_id,
            "account_id": account_id,
            "amount": amount,
            "note": "Other funding accounts for this goal were left unchanged.",
        })
    except Exception as e:
        return json_error("set_goal_contribution", e)
