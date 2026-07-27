"""Re-export of the centralized Settings for backward compatibility.

New code should import from `core.config` directly.
"""

from core.config import settings, Settings  # noqa: F401
