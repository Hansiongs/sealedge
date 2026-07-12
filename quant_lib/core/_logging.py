"""Core logging helpers (console + structured log).
"""

import logging
from rich.console import Console

console = Console()
log = logging.getLogger("rich")
