#!/usr/bin/env python3
"""
LLM Bundle Generator for Reflex

Reads reflex_repo_spec_v1.txt and generates profile-specific text bundles
optimized for LLM context loading.

Usage:
    python reflex_forge_llm_bundle.py <profile_name>
    python reflex_forge_llm_bundle.py overview
    python reflex_forge_llm_bundle.py science_chronome

Output:
    reflex_llm_bundle_<profile>.txt
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Set


def parse_spec(spec_path: Path) -> Dict:
    """Parse the reflex_repo_spec_v1.txt capsule file."""
    with open(spec_path) as f:
        content = f.read()

    spec = {
        'groups': {},
        'profiles': {},
    }

    # Parse [groups] section
    groups_match = re.search(r'\[groups\](.*?)(?=\[profiles\])', content, re.DOTALL)
    if groups_match:
        groups_text = groups_match.group(1)
        # Match group_name = [ ... ]
        for match in re.finditer(r'(\w+)\s*=\s*\[(.*?)\]', groups_text, re.DOTALL):
            group_name = match.group(1)
            files_text = match.group(2)
            # Extract quoted filenames
            files = re.findall(r'"([^"]+)"', files_text)
            spec['groups'][group_name] = files

    # Parse [profiles] section
    profiles_match = re.search(r'\[profiles\](.*?)(?=\[fn |$)', content, re.DOTALL)
    if profiles_match:
        profiles_text = profiles_match.group(1)
        # Match profile_name = { ... }
        for match in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}', profiles_text, re.DOTALL):
            profile_name = match.group(1)
            profile_body = match.group(2)

            profile = {'include': [], 'extra': [], 'exclude': []}

            # Extract include groups
            include_match = re.search(r'include\s*=\s*\[(.*?)\]', profile_body, re.DOTALL)
            if include_match:
                profile['include'] = re.findall(r'"([^"]+)"', include_match.group(1))

            # Extract extra files
            extra_match = re.search(r'extra\s*=\s*\[(.*?)\]', profile_body, re.DOTALL)
            if extra_match:
                profile['extra'] = re.findall(r'"([^"]+)"', extra_match.group(1))

            # Extract exclude groups
            exclude_match = re.search(r'exclude\s*=\s*\[(.*?)\]', profile_body, re.DOTALL)
            if exclude_match:
                profile['exclude'] = re.findall(r'"([^"]+)"', exclude_match.group(1))

            spec['profiles'][profile_name] = profile

    return spec


def resolve_profile_files(spec: Dict, profile_name: str) -> List[str]:
    """Resolve a profile to a flat list of file paths."""
    if profile_name not in spec['profiles']:
        raise ValueError(f"Profile '{profile_name}' not found in spec")

    profile = spec['profiles'][profile_name]
    files: Set[str] = set()

    # Add files from included groups
    for group_name in profile['include']:
        if group_name in spec['groups']:
            files.update(spec['groups'][group_name])
        else:
            print(f"Warning: Group '{group_name}' not found", file=sys.stderr)

    # Remove files from excluded groups
    for group_name in profile['exclude']:
        if group_name in spec['groups']:
            files.difference_update(spec['groups'][group_name])

    # Add extra files directly
    files.update(profile['extra'])

    return sorted(files)


def collapse_blank_lines(text: str) -> str:
    """Collapse multiple consecutive blank lines into a single blank line."""
    # Replace 3+ newlines with exactly 2 newlines (one blank line)
    return re.sub(r'\n\n\n+', '\n\n', text)


def generate_bundle(repo_root: Path, spec: Dict, profile_name: str, output_path: Path):
    """Generate an LLM bundle for the given profile."""
    files = resolve_profile_files(spec, profile_name)

    bundle_parts = []
    bundle_parts.append(f"# Reflex LLM Bundle: {profile_name}")
    bundle_parts.append(f"# Generated from reflex_repo_spec_v1.txt")
    bundle_parts.append(f"# Files included: {len(files)}")
    bundle_parts.append("")

    missing_files = []

    for file_path in files:
        full_path = repo_root / file_path

        if not full_path.exists():
            missing_files.append(file_path)
            continue

        bundle_parts.append("=" * 80)
        bundle_parts.append(f"FILE: {file_path}")
        bundle_parts.append("=" * 80)
        bundle_parts.append("")

        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Collapse blank lines
        content = collapse_blank_lines(content)

        bundle_parts.append(content)
        bundle_parts.append("")

    # Write bundle
    bundle_text = '\n'.join(bundle_parts)
    bundle_text = collapse_blank_lines(bundle_text)  # Final pass

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(bundle_text)

    print(f"Bundle generated: {output_path}")
    print(f"  Files included: {len(files) - len(missing_files)}/{len(files)}")
    if missing_files:
        print(f"  Missing files: {len(missing_files)}")
        for mf in missing_files:
            print(f"    - {mf}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    profile_name = sys.argv[1]
    repo_root = Path(__file__).parent
    spec_path = repo_root / "reflex_repo_spec_v1.txt"
    output_path = repo_root / f"reflex_llm_bundle_{profile_name}.txt"

    if not spec_path.exists():
        print(f"Error: {spec_path} not found", file=sys.stderr)
        sys.exit(1)

    spec = parse_spec(spec_path)

    if profile_name not in spec['profiles']:
        print(f"Error: Profile '{profile_name}' not found", file=sys.stderr)
        print(f"Available profiles: {', '.join(spec['profiles'].keys())}")
        sys.exit(1)

    generate_bundle(repo_root, spec, profile_name, output_path)


if __name__ == '__main__':
    main()
