#!/usr/bin/env python3
"""End-to-end smoke test for monarch-mcp-server over real MCP stdio.

Spawns ``.venv/bin/monarch-mcp-server`` the way Claude Desktop does (the
command plus an env block, no args), initialises it, lists its tools and calls
every read-only tool once with small, safe arguments.

Output is shapes, counts and redacted error text only. It never prints
balances, amounts, account numbers or masks, tokens, or note text.

Usage (run from the repo root, with the repo's own venv):
  .venv/bin/python scripts/smoke_test.py                    # edit mode, reads only
  .venv/bin/python scripts/smoke_test.py --mode read-only   # MONARCH_MCP_READ_ONLY=1
  .venv/bin/python scripts/smoke_test.py --write-roundtrip  # + one reversible note write
  --server PATH   spawn a different entrypoint (e.g. another venv's monarch-mcp-server)

--write-roundtrip (edit mode only) sets a marker note on ONE recent posted,
unsplit transaction that has no note, reads it back, restores the original and
proves the restored value is byte-identical. It never touches amounts,
categories, splits, tags, budgets, goals, rules or balances.

Exit codes: 0 = every check passed, 1 = at least one check failed,
2 = harness/setup error (server did not start, stale recovery file, ...).
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from monarch_mcp_server.read_only import MUTATING_TOOLS

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / ".venv" / "bin" / "monarch-mcp-server"
CALL_TIMEOUT = timedelta(seconds=90)
MARKER_PREFIX = "zz-claude-smoke-test"
# A write tool whose dry_run path returns before any API call, so calling it is
# a no-op in edit mode and must be an "Unknown tool" error in read-only mode.
WRITE_PROBE = (
    "bulk_categorize_transactions",
    {"transaction_ids": [], "category_id": "smoke-dry-run", "dry_run": True},
)

_LONG_DIGITS = re.compile(r"\d{4,}")
_DECIMAL = re.compile(r"-?\$?\d+\.\d+")
_TOKENISH = re.compile(r"[A-Za-z0-9_\-]{28,}")


def redact(text: Any, limit: int = 160) -> str:
    """Strip anything that could be an amount, id, mask or token."""
    out = str(text)
    out = _TOKENISH.sub("<tok>", out)
    out = _DECIMAL.sub("<num>", out)
    out = _LONG_DIGITS.sub("<n>", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:limit]


def shape(obj: Any, depth: int = 0) -> str:
    """Describe a JSON value by structure only (keys and lengths, no values)."""
    if isinstance(obj, list):
        inner = shape(obj[0], depth + 1) if obj and depth < 1 else ""
        return f"list[{len(obj)}]" + (f" of {inner}" if inner else "")
    if isinstance(obj, dict):
        keys = list(obj.keys())
        parts = []
        for key in keys[:10]:
            value = obj[key]
            if isinstance(value, list):
                parts.append(f"{key}[{len(value)}]")
            elif isinstance(value, dict):
                parts.append(f"{key}{{{len(value)}}}")
            else:
                parts.append(str(key))
        more = f",+{len(keys) - 10}" if len(keys) > 10 else ""
        return redact("{" + ",".join(parts) + more + "}", 200)
    return type(obj).__name__


def evaluate(result: Any) -> Tuple[bool, str, Any]:
    """Turn a CallToolResult into (ok, one-line note, parsed payload)."""
    text = "".join(
        getattr(c, "text", "") for c in result.content if getattr(c, "type", "") == "text"
    )
    if result.isError:
        return False, "isError: " + redact(text), None
    if not text.strip():
        return False, "empty response", None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        first = text.strip().splitlines()[0]
        bad = first.startswith(("Error", "❌")) or "failed" in first.lower()
        return (not bad), f"text({len(text)}ch): {redact(first, 90)}", None
    if isinstance(payload, dict):
        if payload.get("error") is True:
            return False, "error: " + redact(payload.get("message")), payload
        if payload.get("success") is False:
            detail = payload.get("errors") or payload.get("error")
            return False, "rejected: " + redact(json.dumps(detail, default=str)), payload
    return True, shape(payload), payload


class Runner:
    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self.rows: List[Dict[str, Any]] = []

    def record(self, tool: str, status: str, note: str) -> None:
        self.rows.append({"tool": tool, "status": status, "note": note})
        print(f"  {status:<4} {tool:<34} {note}", flush=True)

    async def call(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        try:
            result = await self.session.call_tool(tool, args, read_timeout_seconds=CALL_TIMEOUT)
        except Exception as exc:  # transport error / timeout
            self.record(tool, "FAIL", f"transport: {type(exc).__name__}: {redact(exc)}")
            return False, None
        ok, note, payload = evaluate(result)
        self.record(tool, "PASS" if ok else "FAIL", note)
        return ok, payload

    def skip(self, tool: str, why: str) -> None:
        self.record(tool, "SKIP", why)


def _dates() -> Dict[str, str]:
    today = date.today()
    month_start = today.replace(day=1)
    last_day = calendar.monthrange(today.year, today.month)[1]
    prev_month_start = (month_start - timedelta(days=1)).replace(day=1)
    return {
        "today": today.isoformat(),
        "d30": (today - timedelta(days=30)).isoformat(),
        "month_start": month_start.isoformat(),
        "month_end": today.replace(day=last_day).isoformat(),
        "prev_month_start": prev_month_start.isoformat(),
        "q_start": (month_start - timedelta(days=80)).replace(day=1).isoformat(),
    }


def _pick_account(accounts: List[Dict[str, Any]], types: Tuple[str, ...]) -> Optional[str]:
    live = [a for a in accounts if a.get("is_active") and not a.get("is_hidden")]
    for pool in (live, accounts):
        for acct in pool:
            if acct.get("type") in types:
                return acct.get("id")
    return (live or accounts or [{}])[0].get("id")


async def run_reads(r: Runner) -> Dict[str, Any]:
    """Call every read-only tool once. Returns ids useful to later steps."""
    d = _dates()
    ctx: Dict[str, Any] = {}

    for tool in ("setup_authentication", "check_auth_status", "debug_session_loading",
                 "monarch_whoami"):
        await r.call(tool, {})

    _, accounts = await r.call("get_accounts", {})
    accounts = accounts if isinstance(accounts, list) else []
    brokerage = _pick_account(accounts, ("brokerage",))
    depository = _pick_account(accounts, ("depository",))
    for tool, acct in (("get_account_holdings", brokerage),
                       ("get_account_balance_history", depository)):
        if acct:
            await r.call(tool, {"account_id": acct})
        else:
            r.skip(tool, "no account available")

    _, txns = await r.call("get_transactions", {"limit": 25, "start_date": d["d30"],
                                                "end_date": d["today"]})
    rows = (txns or {}).get("data") or [] if isinstance(txns, dict) else []
    ctx["txn_rows"] = rows
    txn_id = rows[0]["id"] if rows else None
    merchant_id = next((x["merchant_id"] for x in rows if x.get("merchant_id")), None)

    await r.call("search_transactions", {"limit": 5, "start_date": d["d30"],
                                         "end_date": d["today"]})
    for tool in ("get_transaction_details", "get_transaction_splits"):
        if txn_id:
            await r.call(tool, {"transaction_id": txn_id})
        else:
            r.skip(tool, "no recent transaction")
    await r.call("get_recurring_transactions", {"start_date": d["month_start"],
                                                "end_date": d["month_end"]})
    await r.call("get_transactions_needing_review", {"limit": 5, "days": 30})
    await r.call("get_transactions_summary", {})
    await r.call("get_spending_summary", {"start_date": d["d30"], "end_date": d["today"]})
    await r.call("get_transaction_tags", {})
    await r.call("get_transaction_rules", {})

    _, cats = await r.call("get_transaction_categories", {})
    await r.call("get_transaction_category_groups", {})
    category_id = next((x["category_id"] for x in rows if x.get("category_id")), None)
    if not category_id and isinstance(cats, list) and cats:
        category_id = cats[0].get("id")
    if category_id:
        await r.call("get_category_details", {"category_id": category_id,
                                              "month": d["month_start"]})
    else:
        r.skip("get_category_details", "no category id")
    await r.call("get_cashflow_by_month", {"start_date": d["prev_month_start"],
                                           "end_date": d["month_end"]})

    await r.call("get_budgets", {"start_date": d["month_start"], "end_date": d["month_end"]})
    await r.call("get_cashflow", {"start_date": d["d30"], "end_date": d["today"]})
    await r.call("get_net_worth", {"start_date": d["d30"], "end_date": d["today"]})
    await r.call("get_net_worth_by_account_type", {"start_date": d["q_start"],
                                                   "timeframe": "month"})
    await r.call("get_account_sync_health", {})
    await r.call("get_debt_paydown", {})

    _, goals = await r.call("get_goals", {})
    goal_list = (goals or {}).get("goals") or [] if isinstance(goals, dict) else []
    if goal_list:
        await r.call("get_goal_contributions", {"goal_id": goal_list[0]["id"],
                                                "month": d["month_start"][:7]})
    else:
        r.skip("get_goal_contributions", "no goals on this account")
    if merchant_id:
        await r.call("get_merchant", {"merchant_id": merchant_id})
    else:
        r.skip("get_merchant", "no merchant id on recent transactions")
    return ctx


READ_TOOLS_PLANNED = frozenset({
    "setup_authentication", "check_auth_status", "debug_session_loading", "monarch_whoami",
    "get_accounts", "get_account_holdings", "get_account_balance_history",
    "get_transactions", "search_transactions", "get_transaction_details",
    "get_transaction_splits", "get_recurring_transactions",
    "get_transactions_needing_review", "get_transactions_summary", "get_spending_summary",
    "get_transaction_tags", "get_transaction_rules", "get_transaction_categories",
    "get_transaction_category_groups", "get_category_details", "get_cashflow_by_month",
    "get_budgets", "get_cashflow", "get_net_worth", "get_net_worth_by_account_type",
    "get_account_sync_health", "get_debt_paydown", "get_goals", "get_goal_contributions",
    "get_merchant",
})


def check_inventory(r: Runner, names: set, mode: str) -> None:
    """Assert the registered tool set matches the mode."""
    writers_present = sorted(names & MUTATING_TOOLS)
    readers = names - MUTATING_TOOLS
    unplanned = sorted(readers - READ_TOOLS_PLANNED)
    missing = sorted(READ_TOOLS_PLANNED - readers)
    if mode == "read-only":
        ok = not writers_present and not missing
        note = f"{len(names)} tools; writers present={len(writers_present)}"
    else:
        absent = sorted(MUTATING_TOOLS - names)
        ok = not absent and not missing
        note = f"{len(names)} tools; writers {len(writers_present)}/{len(MUTATING_TOOLS)}"
        if absent:
            note += f"; ABSENT writers={absent}"
    if writers_present and mode == "read-only":
        note += f"; LEAKED={writers_present}"
    if missing:
        note += f"; missing readers={missing}"
    if unplanned:
        note += f"; readers not in plan={unplanned}"
    r.record("<inventory>", "PASS" if ok else "FAIL", note)


async def write_probe(r: Runner, mode: str) -> None:
    tool, args = WRITE_PROBE
    result = await r.session.call_tool(tool, args, read_timeout_seconds=CALL_TIMEOUT)
    text = "".join(getattr(c, "text", "") for c in result.content)
    if mode == "read-only":
        ok = bool(result.isError) and "unknown tool" in text.lower()
        r.record(f"<probe {tool}>", "PASS" if ok else "FAIL",
                 "refused as unknown tool" if ok else "NOT refused: " + redact(text))
    else:
        ok = not result.isError and '"dry_run": true' in text
        r.record(f"<probe {tool}>", "PASS" if ok else "FAIL",
                 "callable (dry_run no-op)" if ok else redact(text))


def _txn(payload: Any) -> Dict[str, Any]:
    return (payload or {}).get("getTransaction") or {} if isinstance(payload, dict) else {}


def _diff_paths(a: Any, b: Any, prefix: str = "") -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            out += _diff_paths(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
        return out
    return [] if a == b else [prefix or "<root>"]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()


_NULL_NOTES_MUTATION = """
mutation Web_TransactionDrawerUpdateTransaction($input: UpdateTransactionMutationInput!) {
  updateTransaction(input: $input) {
    transaction { id notes __typename }
    errors { message code __typename }
    __typename
  }
}
"""


async def _direct_null_restore(txn_id: str) -> bool:
    """Put a note back to null directly, for servers that predate the fix.

    Before update_transaction_notes(notes="") sent an explicit null, a clear
    through MCP stored "" instead. Monarch treats "" as "no notes" for the
    hasNotes filter, but it is not the original value, so against such a
    server the harness restores null with the server's own session.
    """
    from gql import gql

    from monarch_mcp_server.secure_session import secure_session

    client = secure_session.get_authenticated_client()
    if client is None:
        return False
    result = await client.gql_call(
        operation="Web_TransactionDrawerUpdateTransaction",
        graphql_query=gql(_NULL_NOTES_MUTATION),
        variables={"input": {"id": txn_id, "notes": None}},
    )
    payload = result.get("updateTransaction") or {}
    return not payload.get("errors") and (payload.get("transaction") or {}).get("notes") is None


async def note_roundtrip(r: Runner, rows: List[Dict[str, Any]], recovery: Path) -> None:
    """Set a marker note on one transaction, read it back, restore it exactly."""
    if recovery.exists():
        raise SystemExit(f"stale recovery file {recovery}: a previous round trip did not "
                         "finish. Restore that transaction's note first, then delete it.")
    pick = next((x for x in rows if x.get("notes") in (None, "")
                 and not x.get("is_pending") and not x.get("is_split_transaction")), None)
    if not pick:
        r.skip("<roundtrip>", "no posted, unsplit, note-less recent transaction")
        return
    txn_id = pick["id"]
    ok, before = await r.call("get_transaction_details", {"transaction_id": txn_id})
    if not ok or not _txn(before):
        r.record("<roundtrip>", "FAIL", "could not read the transaction before writing")
        return
    original = _txn(before).get("notes")
    marker = f"{MARKER_PREFIX} {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    fd = os.open(recovery, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump({"transaction_id": txn_id, "original_notes": original,
                   "marker": marker}, fh)
    try:
        ok, wrote = await r.call("update_transaction_notes",
                                 {"transaction_id": txn_id, "notes": marker})
        echoed = (((wrote or {}).get("updateTransaction") or {}).get("transaction") or {})
        r.record("<roundtrip set>", "PASS" if ok and echoed.get("notes") == marker else "FAIL",
                 "mutation echoed the marker" if echoed.get("notes") == marker
                 else "mutation did not echo the marker")
        _, mid = await r.call("get_transaction_details", {"transaction_id": txn_id})
        seen = _txn(mid).get("notes")
        r.record("<roundtrip readback>", "PASS" if seen == marker else "FAIL",
                 "marker present on re-read" if seen == marker else "marker NOT present")
    finally:
        ok, _ = await r.call("update_transaction_notes",
                             {"transaction_id": txn_id, "notes": original or ""})
        _, after = await r.call("get_transaction_details", {"transaction_id": txn_id})
        now = _txn(after).get("notes")
        if original is None and now == "":
            r.record("<roundtrip mcp-restore>", "WARN",
                     "server stored '' not null (pre-fix server); restoring null directly")
            direct_ok = await _direct_null_restore(txn_id)
            r.record("<roundtrip null-restore>", "PASS" if direct_ok else "FAIL",
                     "direct updateTransaction(notes: null) accepted" if direct_ok
                     else "direct null restore rejected")
            _, after = await r.call("get_transaction_details", {"transaction_id": txn_id})
            now = _txn(after).get("notes")
        identical = _digest(now) == _digest(original)
        drift = [p for p in _diff_paths(_txn(before), _txn(after))
                 if not p.startswith("merchant.transactionCount")]
        note = (f"original={type(original).__name__} restored={type(now).__name__} "
                f"sha256 {_digest(original)[:12]}=={_digest(now)[:12]}"
                if identical else
                f"MISMATCH original={type(original).__name__} restored={type(now).__name__}")
        if drift:
            note += f"; other fields changed: {drift}"
        r.record("<roundtrip restore>", "PASS" if identical and not drift else "FAIL", note)
        if identical:
            recovery.unlink()


async def main_async(args: argparse.Namespace) -> int:
    env = {} if args.mode == "edit" else {"MONARCH_MCP_READ_ONLY": "1"}
    params = StdioServerParameters(command=args.server, args=[], env=env)
    log_path = Path(args.server_log)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    print(f"monarch-mcp-server smoke test  mode={args.mode}  "
          f"at={datetime.now().isoformat(timespec='seconds')}")
    print(f"  server stderr -> {log_path} (0600)")
    with os.fdopen(fd, "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                print(f"  server: {init.serverInfo.name} {init.serverInfo.version} "
                      f"protocol={init.protocolVersion}")
                listed = await session.list_tools()
                names = {t.name for t in listed.tools}
                r = Runner(session)
                check_inventory(r, names, args.mode)
                await write_probe(r, args.mode)
                ctx = await run_reads(r)
                if args.write_roundtrip:
                    if args.mode != "edit":
                        r.skip("<roundtrip>", "write round trip needs --mode edit")
                    else:
                        await note_roundtrip(r, ctx.get("txn_rows") or [],
                                             Path(args.recovery_file))
    counts = {s: sum(1 for x in r.rows if x["status"] == s)
              for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print(f"  summary: {counts}")
    if args.json_out:
        out = Path(args.json_out)
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({"mode": args.mode, "tool_count": len(names), "rows": r.rows,
                       "summary": counts}, fh, indent=2)
    return 1 if counts["FAIL"] else 0


def main() -> int:
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("edit", "read-only"), default="edit")
    parser.add_argument("--write-roundtrip", action="store_true")
    parser.add_argument("--json-out", help="write the redacted results here (0600)")
    parser.add_argument("--server-log", default=str(tmp / "monarch-smoke-server.log"))
    parser.add_argument("--recovery-file", default=str(tmp / "monarch-smoke-roundtrip.json"))
    parser.add_argument("--server", default=str(SERVER),
                        help="server entrypoint to spawn (default: this repo's .venv)")
    args = parser.parse_args()
    if not Path(args.server).exists():
        print(f"server entrypoint missing: {args.server}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(main_async(args))
    except SystemExit as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
