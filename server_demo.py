from fastmcp import FastMCP
import json
from typing import Dict, List, Any

mcp = FastMCP("Demo")

@mcp.tool(name="json.read")
def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file and return its object."""
    with open(path) as f:
        return json.load(f)

@mcp.tool(name="math.add")
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b

@mcp.tool(name="transform.filter_keys")
def filter_keys(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Return a dict containing only selected keys."""
    return {k: data[k] for k in keys if k in data}

if __name__ == "__main__":
    mcp.run()