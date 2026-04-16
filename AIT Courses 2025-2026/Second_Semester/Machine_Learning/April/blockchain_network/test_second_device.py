"""Simple test client for a blockchain node running on another device.

Default targets:
- http://192.168.0.182:5001
- http://192.168.0.137:5002

What it does:
- checks /status
- checks /chain
- posts a sample transaction
- optionally mines a block
- traces a transaction if you pass a tx_id
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_BASE_URLS = [
    "http://192.168.0.182:5001",
    "http://192.168.0.137:5002",
]


def request_json(url: str, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    with urllib_request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def safe_request(url: str, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        return {"ok": True, "data": request_json(url, method=method, payload=payload)}
    except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


def run_checks(base_url: str, args: argparse.Namespace) -> None:
    base_url = base_url.rstrip("/")

    print(f"\n===== TARGET {base_url} =====")

    print("== STATUS ==")
    print(json.dumps(safe_request(f"{base_url}/status"), indent=2, ensure_ascii=False))

    print("\n== CHAIN ==")
    print(json.dumps(safe_request(f"{base_url}/chain"), indent=2, ensure_ascii=False))

    if args.register_peer:
        print("\n== REGISTER PEER ==")
        payload = {"nodes": [args.register_peer]}
        print(json.dumps(safe_request(f"{base_url}/nodes/register", method="POST", payload=payload), indent=2, ensure_ascii=False))

    print("\n== SEND TRANSACTION ==")
    tx_payload = {
        "sender": args.sender,
        "recipient": args.recipient,
        "amount": args.amount,
        "memo": args.memo,
        "broadcast": False,
    }
    tx_result = safe_request(f"{base_url}/transactions", method="POST", payload=tx_payload)
    print(json.dumps(tx_result, indent=2, ensure_ascii=False))

    tx_id = args.tx_id
    if not tx_id and tx_result.get("ok"):
        tx_id = tx_result["data"]["transaction"]["tx_id"]

    if tx_id:
        print("\n== TRACE TRANSACTION ==")
        print(json.dumps(safe_request(f"{base_url}/transactions/{tx_id}/trace"), indent=2, ensure_ascii=False))

    if args.mine:
        print("\n== MINE BLOCK ==")
        mine_payload = {"miner": args.miner}
        print(json.dumps(safe_request(f"{base_url}/mine", method="POST", payload=mine_payload), indent=2, ensure_ascii=False))

    print("\n== RESOLVE CONSENSUS ==")
    print(json.dumps(safe_request(f"{base_url}/nodes/resolve"), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a remote blockchain node")
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        default=[],
        help="remote node URL; can be passed multiple times",
    )
    parser.add_argument("--sender", default="alice", help="transaction sender")
    parser.add_argument("--recipient", default="bob", help="transaction recipient")
    parser.add_argument("--amount", type=float, default=3.5, help="transaction amount")
    parser.add_argument("--memo", default="device test", help="transaction memo")
    parser.add_argument("--miner", default="tester-node", help="miner reward address for /mine")
    parser.add_argument("--mine", action="store_true", help="mine a block after the transaction test")
    parser.add_argument("--tx-id", default="", help="trace an existing transaction ID")
    parser.add_argument("--register-peer", default="", help="optional peer URL to register on the remote node")
    args = parser.parse_args()

    base_urls = args.base_urls or DEFAULT_BASE_URLS

    for base_url in base_urls:
        run_checks(base_url, args)


if __name__ == "__main__":
    main()
