#!/usr/bin/env python3
"""Entry point for running ickyMCP server."""

import asyncio
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from src.server import main

if __name__ == "__main__":
    asyncio.run(main())
