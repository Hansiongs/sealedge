# `quant_lib.core`

Core simulation engine, WFA, statistics, and risk allocation.

This is the **private** implementation layer. Most users should use
`quant_lib.tools` (composable public API) or `quant_lib.research`
(high-level workflow). Direct imports from `quant_lib.core` are
supported but may break across minor versions.

Sprint 2 fix: rendered members = summary (signatures + docstrings) so
the API reference is discoverable. Use `quant_lib.tools` for the
recommended public API.

::: quant_lib.core
    options:
      show_root_heading: false
      show_source: false
      members: summary
      docstring_style: numpy
