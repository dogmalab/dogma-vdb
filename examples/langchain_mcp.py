#!/usr/bin/env python3
"""Example: using dogma-vdb MCP server from LangChain.

Prerequisites:
  pip install langchain langchain-mcp-adapters

Usage:
  1. Build the MCP server: cargo build --release -p dogma-vdb-mcp
  2. Run this script: python examples/langchain_mcp.py

The script starts dogma-mcp server as a subprocess and connects
LangChain to it via the MCP stdio adapter.  All vector operations
(ingest, query, delete) are exposed as LangChain tools.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal MCP client — no LangChain dependency required
# ---------------------------------------------------------------------------

async def demo_without_langchain():
    """Use dogma-vdb through raw MCP stdio (no LangChain needed)."""
    print("=" * 60)
    print("Demo: dogma-vdb via raw MCP stdio")
    print("=" * 60)

    # Find the MCP binary
    mcp_bin = (
        Path(__file__).resolve().parent.parent
        / "target" / "release" / "dogma-vdb-mcp"
    )
    if not mcp_bin.exists():
        mcp_bin = (
            Path(__file__).resolve().parent.parent
            / "dogma-vdb-mcp" / "target" / "release" / "dogma-vdb-mcp"
        )
    if not mcp_bin.exists():
        print("Build the MCP server first: cargo build --release -p dogma-vdb-mcp")
        sys.exit(1)

    # Start MCP server as subprocess
    proc = await asyncio.create_subprocess_exec(
        str(mcp_bin),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    async def send(msg: dict) -> dict:
        """Send a JSON-RPC message and read the response."""
        line = json.dumps(msg) + "\n"
        proc.stdin.write(line.encode())
        await proc.stdin.drain()
        resp = await proc.stdout.readline()
        return json.loads(resp)

    async def send_request(method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request."""
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        resp = await send(req)
        print(f"  {method} → {json.dumps(resp.get('result', resp.get('error')), indent=2)[:200]}")
        return resp

    # 1. Initialize
    await send_request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "demo", "version": "1.0"},
    })

    # 2. List available tools
    tools_resp = await send_request("tools/list")
    tools = tools_resp.get("result", {}).get("tools", [])
    print(f"\n  Available tools ({len(tools)}):")
    for t in tools:
        print(f"    - {t['name']}: {t.get('description', '')[:60]}")

    # 3. Ingest a document (embedding for "Rust is safe" with all-MiniLM-L6-v2)
    sample_emb = [0.01] * 384  # dummy — replace with real embedding
    await send_request("tools/call", {
        "name": "vecdb_ingest",
        "arguments": {
            "path": "/tmp/demo_langchain.vdb",
            "id": "doc-1",
            "text": "Rust is safe and fast",
            "embedding": sample_emb,
            "metadata": {"lang": "en", "topic": "rust"},
        },
    })

    # 4. Query
    await send_request("tools/call", {
        "name": "vecdb_query",
        "arguments": {
            "path": "/tmp/demo_langchain.vdb",
            "query": sample_emb,
            "k": 5,
        },
    })

    # 5. Collection info
    await send_request("tools/call", {
        "name": "vecdb_info",
        "arguments": {"path": "/tmp/demo_langchain.vdb"},
    })

    proc.terminate()
    print("\n  ✅ Demo complete\n")


# ---------------------------------------------------------------------------
# LangChain integration (optional — requires langchain-mcp-adapters)
# ---------------------------------------------------------------------------

async def demo_with_langchain():
    """Use dogma-vdb through LangChain's MCP adapter."""
    try:
        from langchain_mcp_adapters.client import StdioClient
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_openai import ChatOpenAI
    except ImportError:
        print(
            "Skipping LangChain demo — install with:\n"
            "  pip install langchain langchain-mcp-adapters langchain-openai"
        )
        return

    print("=" * 60)
    print("Demo: dogma-vdb via LangChain MCP adapter")
    print("=" * 60)

    mcp_bin = (
        Path(__file__).resolve().parent.parent
        / "target" / "release" / "dogma-vdb-mcp"
    )

    async with StdioClient([str(mcp_bin)]) as client:
        tools = await client.list_tools()
        print(f"  Connected to dogma-vdb MCP — {len(tools)} tools loaded")

        # Use tools directly (no agent needed for basic usage)
        result = await client.call_tool(
            "vecdb_ingest",
            {
                "path": "/tmp/demo_langchain.vdb",
                "id": "doc-2",
                "text": "Python is easy to write",
                "embedding": [0.02] * 384,
            },
        )
        print(f"  Ingest result: {result}")

        result = await client.call_tool(
            "vecdb_query",
            {"path": "/tmp/demo_langchain.vdb", "query": [0.01] * 384, "k": 5},
        )
        print(f"  Query result: {result}")

    print("  ✅ LangChain demo complete\n")


async def main():
    await demo_without_langchain()
    await demo_with_langchain()


if __name__ == "__main__":
    asyncio.run(main())
