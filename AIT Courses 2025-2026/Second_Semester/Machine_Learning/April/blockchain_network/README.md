# AIT Blockchain Node

This folder contains a small educational blockchain prototype that can run on several devices over a network.

## What it does

- creates transactions with unique IDs
- mines blocks with proof-of-work
- traces a transaction through the chain
- registers peer nodes on the network
- broadcasts transactions and mined blocks to peers
- resolves conflicts by keeping the longest valid chain

## Run

Install Flask first:

```bash
pip install flask
```

Start one node:

```bash
python blockchain.py --port 5000 --node-id node-a
```

Start another node on a different device or port:

```bash
python blockchain.py --port 5001 --node-id node-b --peer http://192.168.1.10:5000
```

Register peers after startup:

```bash
curl -X POST http://127.0.0.1:5000/nodes/register -H "Content-Type: application/json" -d "{\"nodes\":[\"http://192.168.1.11:5001\"]}"
```

## Useful endpoints

- `GET /status` - node health and chain length
- `GET /chain` - full chain snapshot
- `POST /transactions` - create a new transaction
- `GET /transactions/<tx_id>/trace` - trace a transaction
- `POST /mine` - mine a new block
- `POST /nodes/register` - add peers
- `GET /nodes/resolve` - run consensus

## Security note

This is a learning prototype, not a production blockchain. For real deployments you should add:

- TLS between nodes
- signed transactions with public/private keys
- authenticated peer registration
- persistence on disk or a database
- anti-spam and rate limiting
- stronger consensus than simple longest-chain replacement
