"""Tests for ACE aliases used by Transformers' docstring generation."""

import importlib
import unittest

from acestep.models.common.transformers_docstrings import (
    register_acestep_docstring_configs,
)


class TransformersDocstringsTest(unittest.TestCase):
    """Verify custom ACE module names resolve without a console warning."""

    def test_registers_xl_base_config_alias(self) -> None:
        """Associate XL-base documentation with the ACE config class."""
        module = importlib.import_module("transformers.utils.auto_docstring")
        module.HARDCODED_CONFIG_FOR_MODELS.pop("acestep-v15-xl-base", None)
        register_acestep_docstring_configs()
        self.assertEqual(
            module.HARDCODED_CONFIG_FOR_MODELS["acestep-v15-xl-base"],
            "AceStepConfig",
        )


if __name__ == "__main__":
    unittest.main()
