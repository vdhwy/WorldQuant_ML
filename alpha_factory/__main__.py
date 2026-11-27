"""Module entrypoint enabling ``python -m alpha_factory <subcommand>``."""

from __future__ import annotations

import sys

from alpha_factory.ops.cli import main

if __name__ == "__main__":
    sys.exit(main())
