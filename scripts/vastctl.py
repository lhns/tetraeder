"""Shim: the Task Scheduler commands still call this file; the tool is now taxameter (ADR 0029)."""
import sys

from taxameter.cli import main

sys.exit(main())
