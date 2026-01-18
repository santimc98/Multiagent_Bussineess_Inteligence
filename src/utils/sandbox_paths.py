"""
Universal sandbox path standardization.

Provides canonical paths and alias handling for sandbox inputs.
This ensures that LLM-generated code works regardless of which path name it uses.
"""

from typing import List

# Canonical paths relative to run_root
CANONICAL_RAW_REL = "data/raw.csv"
CANONICAL_CLEANED_REL = "data/cleaned_data.csv"
CANONICAL_MANIFEST_REL = "data/cleaning_manifest.json"

# Common aliases that LLMs might use for raw data
COMMON_RAW_ALIASES = [
    "raw_data.csv",
    "input.csv",
    "data/raw_data.csv",
    "data/input.csv",
    "data/data.csv",
]

# Common aliases that LLMs might use for cleaned data
# Note: We include raw_data/input as cleaned aliases to be lenient with junior LLMs
COMMON_CLEANED_ALIASES = [
    "data.csv",
    "cleaned.csv",
    "cleaned_data.csv",
    "train.csv",
    "test.csv",
    "raw_data.csv",  # Lenient: some LLMs confuse naming
    "input.csv",  # Lenient: some LLMs confuse naming
    "data/raw_data.csv",
    "data/input.csv",
    "data/data.csv",
    "data/train.csv",
    "data/test.csv",
]


def patch_placeholders(code: str, *, data_rel: str = CANONICAL_CLEANED_REL, manifest_rel: str = CANONICAL_MANIFEST_REL) -> str:
    """
    Replace ONLY universal placeholders in code.

    This is a minimal, focused replacement that doesn't do massive path patching.

    Args:
        code: The code to patch
        data_rel: Relative path to use for $data_path (default: CANONICAL_CLEANED_REL)
        manifest_rel: Relative path to use for $manifest_path (default: CANONICAL_MANIFEST_REL)

    Returns:
        Patched code with placeholders replaced
    """

    # Replace universal placeholders
    code = code.replace("$data_path", data_rel)
    code = code.replace("${data_path}", data_rel)
    code = code.replace("$manifest_path", manifest_rel)
    code = code.replace("${manifest_path}", manifest_rel)

    return code


def build_symlink_or_copy_commands(run_root: str, *, canonical_rel: str, aliases: List[str]) -> List[str]:
    """
    Build shell commands to create aliases for a canonical path.

    For each alias, creates either a symlink or a fallback copy.

    Args:
        run_root: The sandbox run root directory
        canonical_rel: Canonical relative path (e.g., "data/cleaned_data.csv")
        aliases: List of alias paths (can be relative to run_root or nested in data/)

    Returns:
        List of shell commands to execute
    """
    commands = []

    # Ensure data directory exists
    commands.append(f"mkdir -p {run_root}/data")

    # Separate root aliases and data/ aliases
    # Root aliases: paths without "/" or starting with "data/" (for compatibility)
    # Data aliases: paths starting with "data/" (create inside data/ directory)
    root_aliases = [a for a in aliases if "/" not in a or not a.startswith("data/")]
    data_aliases = [a for a in aliases if a.startswith("data/")]

    # Create root aliases from run_root
    if root_aliases:
        commands.append(f"cd {run_root}")
        canonical_in_data = f"data/{canonical_rel.split('/')[-1]}"
        for alias in root_aliases:
            commands.append(
                f"ln -sf {canonical_in_data} {alias} || "
                f"cp -f {canonical_in_data} {alias}"
            )

    # Create data/ aliases from run_root/data
    if data_aliases:
        commands.append(f"cd {run_root}/data")
        canonical_basename = canonical_rel.split("/")[-1]
        for alias in data_aliases:
            alias_basename = alias.split("/")[-1]
            commands.append(
                f"ln -sf {canonical_basename} {alias_basename} || "
                f"cp -f {canonical_basename} {alias_basename}"
            )

    return commands


def canonical_abs(run_root: str, rel_path: str) -> str:
    """
    Safely join run_root and relative path, avoiding double slashes.

    Args:
        run_root: The sandbox run root directory
        rel_path: Relative path (e.g., "data/cleaned_data.csv")

    Returns:
        Absolute path without double slashes
    """
    return f"{run_root.rstrip('/')}/{rel_path.lstrip('/')}"
