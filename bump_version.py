#!/usr/bin/env python3
"""
Automates version bumping across the Dedekind repository.
Updates:
  - pyproject.toml
  - src/dedekind/__init__.py
  - README.md (release info)
  - docs/Dedekind_Language_Specification.md (specification status version)

Usage:
  python bump_version.py <new_version>
Example:
  python bump_version.py 3.0.2
"""

import sys
import os
import re

def main():
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py <new_version>")
        print("Example: python bump_version.py 3.0.2")
        sys.exit(1)
        
    new_version = sys.argv[1].strip()
    if not re.match(r"^\d+\.\d+\.\d+$", new_version):
        print(f"Error: Version '{new_version}' does not match format X.Y.Z")
        sys.exit(1)
        
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Parse current version from pyproject.toml
    pyproject_path = os.path.join(project_root, "pyproject.toml")
    if not os.path.isfile(pyproject_path):
        print(f"Error: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)
        
    with open(pyproject_path, "r", encoding="utf-8") as f:
        pyproject_content = f.read()
        
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_content, re.MULTILINE)
    if not version_match:
        print("Error: Could not find version key in pyproject.toml")
        sys.exit(1)
        
    old_version = version_match.group(1)
    if old_version == new_version:
        print(f"Version is already {new_version}. Nothing to do.")
        sys.exit(0)
        
    print(f"Bumping version from {old_version} to {new_version}...")
    
    # 2. Update pyproject.toml
    new_pyproject = re.sub(
        r'(^version\s*=\s*")([^"]+)(")',
        rf'\g<1>{new_version}\g<3>',
        pyproject_content,
        flags=re.MULTILINE
    )
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_pyproject)
    print("Updated pyproject.toml")
    
    # 3. Update src/dedekind/__init__.py
    init_path = os.path.join(project_root, "src", "dedekind", "__init__.py")
    if os.path.isfile(init_path):
        with open(init_path, "r", encoding="utf-8") as f:
            init_content = f.read()
        new_init = re.sub(
            r'(__version__\s*=\s*")([^"]+)(")',
            rf'\g<1>{new_version}\g<3>',
            init_content
        )
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(new_init)
        print("Updated src/dedekind/__init__.py")
    else:
        print("Warning: src/dedekind/__init__.py not found")
        
    # 4. Update README.md
    readme_path = os.path.join(project_root, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        
        # Replace the current release version in status section
        new_readme = re.sub(
            r'(- Current release:\s+\*\*v)[^*]+(\*\*)',
            rf'\g<1>{new_version}\g<2>',
            readme_content
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_readme)
        print("Updated README.md")
    else:
        print("Warning: README.md not found")
        
    # 5. Update docs/Dedekind_Language_Specification.md
    spec_path = os.path.join(project_root, "docs", "Dedekind_Language_Specification.md")
    if os.path.isfile(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_content = f.read()
            
        new_spec = spec_content.replace(
            f"updated to **v{old_version}**",
            f"updated to **v{new_version}**"
        )
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(new_spec)
        print("Updated docs/Dedekind_Language_Specification.md (spec status version)")
        print("Note: You may still want to manually add the version entry to the revision list at the top of the spec file.")
    else:
        print("Warning: docs/Dedekind_Language_Specification.md not found")
        
    print(f"Successfully bumped version to {new_version}!")

if __name__ == "__main__":
    main()
