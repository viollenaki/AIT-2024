"""A small educational blockchain node with a network API.

This module provides:
- transaction tracing
- chain validation
- peer registration and chain sync
- optional block broadcasting between devices on the same network

The implementation is intentionally simple and is suitable for demos,
labs, and small multi-device experiments. It is not production-grade
security.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from flask import Flask, jsonify, request as flask_request


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_node_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("node url cannot be empty")
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


def now_ts() -> float:
    return round(time.time(), 6)


def make_transaction(sender: str, recipient: str, amount: float, memo: str = "") -> Dict[str, Any]:
    if not sender or not recipient:
        raise ValueError("sender and recipient are required")
    if amount <= 0:
        raise ValueError("amount must be greater than zero")

    created_at = now_ts()
    payload = {
        "sender": sender,
        "recipient": recipient,
        "amount": amount,
        "memo": memo,
        "created_at": created_at,
    }
    tx_id = sha256_hex(canonical_json(payload) + uuid.uuid4().hex)
    payload["tx_id"] = tx_id
    return payload


class BlockchainNode:
    def __init__(self, node_id: str, difficulty: int = 4) -> None:
        self.node_id = node_id
        self.difficulty = max(1, difficulty)
        self._lock = threading.Lock()
        self._chain: List[Dict[str, Any]] = []
        self._pending_transactions: List[Dict[str, Any]] = []
        self._peers: set[str] = set()
        self._tx_index: set[str] = set()
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        genesis = self._build_block(
            index=1,
            transactions=[],
            previous_hash="1",
            nonce=0,
            timestamp=now_ts(),
        )
        self._chain.append(genesis)

    def _build_block(
        self,
        index: int,
        transactions: List[Dict[str, Any]],
        previous_hash: str,
        nonce: int,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        block = {
            "index": index,
            "timestamp": timestamp if timestamp is not None else now_ts(),
            "transactions": transactions,
            "nonce": nonce,
            "previous_hash": previous_hash,
            "miner": self.node_id,
        }
        block["hash"] = sha256_hex(canonical_json(block))
        return block

    def _mine_proof(
        self,
        index: int,
        transactions: List[Dict[str, Any]],
        previous_hash: str,
    ) -> Dict[str, Any]:
        nonce = 0
        timestamp = now_ts()
        target_prefix = "0" * self.difficulty

        while True:
            candidate = {
                "index": index,
                "timestamp": timestamp,
                "transactions": transactions,
                "nonce": nonce,
                "previous_hash": previous_hash,
                "miner": self.node_id,
            }
            block_hash = sha256_hex(canonical_json(candidate))
            if block_hash.startswith(target_prefix):
                candidate["hash"] = block_hash
                return candidate
            nonce += 1

    def add_peer(self, peer_url: str) -> str:
        normalized = normalize_node_url(peer_url)
        with self._lock:
            self._peers.add(normalized)
        return normalized

    def register_peers(self, peers: List[str]) -> List[str]:
        registered = []
        for peer in peers:
            registered.append(self.add_peer(peer))
        return sorted(set(registered))

    def peers(self) -> List[str]:
        with self._lock:
            return sorted(self._peers)

    def create_transaction(self, sender: str, recipient: str, amount: float, memo: str = "") -> Dict[str, Any]:
        transaction = make_transaction(sender, recipient, amount, memo)
        with self._lock:
            self._pending_transactions.append(transaction)
            self._tx_index.add(transaction["tx_id"])
        return transaction

    def relay_transaction(self, transaction: Dict[str, Any]) -> bool:
        tx_id = transaction.get("tx_id")
        if not tx_id:
            return False
        with self._lock:
            if tx_id in self._tx_index:
                return False
            self._pending_transactions.append(transaction)
            self._tx_index.add(tx_id)
        return True

    def mine_block(self, miner_reward_address: str) -> Dict[str, Any]:
        reward = make_transaction("NETWORK", miner_reward_address, 1.0, "mining reward")

        with self._lock:
            transactions = [*self._pending_transactions, reward]
            self._pending_transactions = []
            self._tx_index.add(reward["tx_id"])
            index = len(self._chain) + 1
            previous_hash = self._chain[-1]["hash"]

        block = self._mine_proof(index=index, transactions=transactions, previous_hash=previous_hash)

        with self._lock:
            self._chain.append(block)

        return block

    def relay_block(self, block: Dict[str, Any]) -> bool:
        with self._lock:
            if self._is_valid_next_block(block, self._chain[-1]):
                self._chain.append(block)
                for tx in block.get("transactions", []):
                    tx_id = tx.get("tx_id")
                    if tx_id:
                        self._tx_index.add(tx_id)
                self._pending_transactions = [
                    tx for tx in self._pending_transactions if tx.get("tx_id") not in self._tx_index
                ]
                return True
        return False

    def _is_valid_next_block(self, block: Dict[str, Any], previous_block: Dict[str, Any]) -> bool:
        required_keys = {"index", "timestamp", "transactions", "nonce", "previous_hash", "miner", "hash"}
        if not required_keys.issubset(block):
            return False
        if block["index"] != previous_block["index"] + 1:
            return False
        if block["previous_hash"] != previous_block["hash"]:
            return False
        candidate = dict(block)
        block_hash = candidate.pop("hash")
        recalculated = sha256_hex(canonical_json(candidate))
        if recalculated != block_hash:
            return False
        return block_hash.startswith("0" * self.difficulty)

    def is_chain_valid(self, chain: Optional[List[Dict[str, Any]]] = None) -> bool:
        data = chain if chain is not None else self._chain
        if not data:
            return False

        for index in range(1, len(data)):
            previous_block = data[index - 1]
            current_block = data[index]
            if not self._is_valid_next_block(current_block, previous_block):
                return False
        return True

    def trace_transaction(self, tx_id: str) -> Dict[str, Any]:
        with self._lock:
            for block in self._chain:
                for position, transaction in enumerate(block["transactions"]):
                    if transaction.get("tx_id") == tx_id:
                        return {
                            "found": True,
                            "status": "confirmed",
                            "block_index": block["index"],
                            "transaction_position": position,
                            "confirmations": len(self._chain) - block["index"] + 1,
                            "block_hash": block["hash"],
                            "previous_hash": block["previous_hash"],
                            "transaction": transaction,
                        }

            for position, transaction in enumerate(self._pending_transactions):
                if transaction.get("tx_id") == tx_id:
                    return {
                        "found": True,
                        "status": "pending",
                        "block_index": None,
                        "transaction_position": position,
                        "confirmations": 0,
                        "block_hash": None,
                        "previous_hash": None,
                        "transaction": transaction,
                    }

        return {"found": False, "status": "missing", "tx_id": tx_id}

    def resolve_conflicts(self) -> Dict[str, Any]:
        best_chain = None
        best_length = len(self._chain)

        for peer in self.peers():
            peer_chain = self._fetch_peer_chain(peer)
            if not peer_chain:
                continue
            if len(peer_chain) > best_length and self.is_chain_valid(peer_chain):
                best_chain = peer_chain
                best_length = len(peer_chain)

        if best_chain is None:
            return {"replaced": False, "length": len(self._chain)}

        with self._lock:
            self._chain = best_chain
            self._pending_transactions = [
                tx
                for tx in self._pending_transactions
                if tx.get("tx_id") not in {item.get("tx_id") for block in self._chain for item in block["transactions"]}
            ]
        return {"replaced": True, "length": len(self._chain)}

    def _fetch_peer_chain(self, peer_url: str) -> Optional[List[Dict[str, Any]]]:
        try:
            with urllib_request.urlopen(f"{peer_url}/chain", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
                chain = payload.get("chain")
                if isinstance(chain, list):
                    return chain
        except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return None
        return None

    def broadcast_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for peer in self.peers():
            results[peer] = self._post_json(f"{peer}/transactions/relay", transaction)
        return results

    def broadcast_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for peer in self.peers():
            results[peer] = self._post_json(f"{peer}/blocks/relay", block)
        return results

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return {"error": str(exc)}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            chain_hash = self._chain[-1]["hash"] if self._chain else None
            return {
                "node_id": self.node_id,
                "difficulty": self.difficulty,
                "chain_length": len(self._chain),
                "pending_transactions": len(self._pending_transactions),
                "peers": sorted(self._peers),
                "latest_hash": chain_hash,
                "chain_valid": self.is_chain_valid(),
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "chain": self._chain,
                "pending_transactions": self._pending_transactions,
                "peers": sorted(self._peers),
                "length": len(self._chain),
                "valid": self.is_chain_valid(),
            }


def create_app(node: BlockchainNode) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Any:
        return jsonify(
            {
                "name": "AIT Blockchain Node",
                "status": node.status(),
                "endpoints": [
                    "/status",
                    "/chain",
                    "/transactions",
                    "/transactions/<tx_id>/trace",
                    "/mine",
                    "/nodes",
                    "/nodes/register",
                    "/nodes/resolve",
                    "/transactions/relay",
                    "/blocks/relay",
                ],
            }
        )

    @app.get("/status")
    def status() -> Any:
        return jsonify(node.status())

    @app.get("/chain")
    def chain() -> Any:
        return jsonify(node.snapshot())

    @app.post("/transactions")
    def transactions() -> Any:
        payload = flask_request.get_json(force=True, silent=True) or {}
        try:
            transaction = node.create_transaction(
                sender=payload["sender"],
                recipient=payload["recipient"],
                amount=float(payload["amount"]),
                memo=str(payload.get("memo", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        broadcast = bool(payload.get("broadcast", True))
        if broadcast:
            transaction["broadcast_results"] = node.broadcast_transaction(transaction)
        return jsonify({"transaction": transaction, "pending": len(node.snapshot()["pending_transactions"])})

    @app.post("/transactions/relay")
    def relay_transaction() -> Any:
        payload = flask_request.get_json(force=True, silent=True) or {}
        if node.relay_transaction(payload):
            return jsonify({"accepted": True, "tx_id": payload.get("tx_id")})
        return jsonify({"accepted": False, "reason": "duplicate or invalid transaction"}), 409

    @app.get("/transactions/<tx_id>/trace")
    def trace(tx_id: str) -> Any:
        return jsonify(node.trace_transaction(tx_id))

    @app.post("/mine")
    def mine() -> Any:
        payload = flask_request.get_json(force=True, silent=True) or {}
        miner_address = str(payload.get("miner", node.node_id))
        block = node.mine_block(miner_address)
        broadcast_results = node.broadcast_block(block)
        return jsonify({"block": block, "broadcast_results": broadcast_results})

    @app.post("/blocks/relay")
    def relay_block() -> Any:
        payload = flask_request.get_json(force=True, silent=True) or {}
        if node.relay_block(payload):
            return jsonify({"accepted": True, "block_index": payload.get("index")})
        return jsonify({"accepted": False, "reason": "block rejected"}), 409

    @app.get("/nodes")
    def nodes() -> Any:
        return jsonify({"peers": node.peers()})

    @app.post("/nodes/register")
    def register_nodes() -> Any:
        payload = flask_request.get_json(force=True, silent=True) or {}
        peers = payload.get("nodes", [])
        if not isinstance(peers, list):
            return jsonify({"error": "nodes must be a list"}), 400
        try:
            registered = node.register_peers([str(peer) for peer in peers])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"registered": registered, "peers": node.peers()})

    @app.get("/nodes/resolve")
    def resolve() -> Any:
        return jsonify(node.resolve_conflicts())

    @app.post("/sync")
    def sync() -> Any:
        payload = flask_request.get_json(force=True, silent=True) or {}
        peers = payload.get("nodes", [])
        if isinstance(peers, list):
            node.register_peers([str(peer) for peer in peers])
        return jsonify(node.resolve_conflicts())

    return app


def build_node_from_args(args: argparse.Namespace) -> BlockchainNode:
    node_id = args.node_id or f"node-{uuid.uuid4().hex[:8]}"
    node = BlockchainNode(node_id=node_id, difficulty=args.difficulty)
    if args.peers:
        node.register_peers(args.peers)
    return node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an educational blockchain node")
    parser.add_argument("--host", default="0.0.0.0", help="host interface to bind")
    parser.add_argument("--port", type=int, default=5000, help="port to bind")
    parser.add_argument("--node-id", default="", help="unique node name")
    parser.add_argument("--difficulty", type=int, default=4, help="proof-of-work difficulty")
    parser.add_argument(
        "--peer",
        dest="peers",
        action="append",
        default=[],
        help="peer node URL; can be passed multiple times",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node = build_node_from_args(args)
    app = create_app(node)
    app.run(host=args.host, port=args.port, debug=True, threaded=True)


if __name__ == "__main__":
    main()
