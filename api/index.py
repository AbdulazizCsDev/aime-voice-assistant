"""
Vercel serverless entry point.

Vercel's @vercel/python builder looks for `app` (ASGI) in files under /api,
so we re-export the FastAPI app defined in server.py at the repo root.
"""

import os
import sys

# Make the repo root importable so we can pull in server.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: E402,F401
