"""Register ACE model aliases used by Transformers' docstring decorator."""

from importlib import import_module


def register_acestep_docstring_configs() -> None:
    """Map ACE module names to their shared config class for documentation."""
    module = import_module("transformers.utils.auto_docstring")
    aliases = (
        "acestep-v15-base",
        "acestep-v15-sft",
        "acestep-v15-turbo",
        "acestep-v15-xl-base",
        "acestep-v15-xl-sft",
        "acestep-v15-xl-turbo",
    )
    for alias in aliases:
        module.HARDCODED_CONFIG_FOR_MODELS.setdefault(alias, "AceStepConfig")


register_acestep_docstring_configs()
auto_docstring = import_module("transformers.utils.auto_docstring").auto_docstring
