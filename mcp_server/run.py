"""Launcher that works from any folder: python D:\\hirepanel\\mcp_server\\run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.server import main  # noqa: E402

main()
