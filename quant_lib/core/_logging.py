"""
Logger setup -- shared console and logger singletons.

Provides ``log`` (the framework logger) and ``console`` (the Rich
console) for use throughout ``quant_lib``. No file handler is
installed at import time: file logging is the CLI's responsibility
(see ``quant_lib.utils.logging.setup_logging``), which writes to
a configurable path. Importing this module is side-effect-free
apart from creating the Console object.
"""

import logging
from rich.console import Console

console = Console()
log = logging.getLogger("rich")
