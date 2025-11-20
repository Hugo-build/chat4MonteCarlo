#!/usr/bin/env python3
"""
Analyze Python files and show which packages are imported.
Useful for generating/updating pyproject.toml

Usage:
    python list_project_dependencies.py
"""

import sys
from pathlib import Path
import re
from collections import defaultdict

# Common import name to package name mappings
PACKAGE_MAP = {
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'pyyaml',
    'dotenv': 'python-dotenv',
    'mpl_toolkits': 'matplotlib',
}

# Standard library modules (don't need to be installed)
STDLIB = {
    'os', 'sys', 'json', 'time', 'pathlib', 'typing', 're', 'subprocess',
    'threading', 'queue', 'argparse', 'io', 'collections', 'itertools',
    'functools', 'copy', 'datetime', 'math', 'random', 'warnings',
    'logging', 'unittest', 'uuid', 'base64', 'hashlib', 'urllib',
    'http', 'email', 'socket', 'struct', 'pickle', 'shelve', 'csv',
    'configparser', 'tempfile', 'shutil', 'glob', 'fnmatch', 'abc',
    'enum', 'dataclasses', 'contextlib', 'inspect', 'importlib',
    'getpass', 'string', 'textwrap', 'pprint', 'difflib', 'platform',
}


def find_imports(project_root: Path):
    """Find all imports in Python files"""
    
    # Find all Python files
    py_files = []
    for pattern in ['*.py', '*/*.py', '*/*/*.py']:
        py_files.extend(project_root.glob(pattern))
    
    # Track imports and where they're used
    imports = defaultdict(list)
    
    for py_file in py_files:
        # Skip virtual environments and cache
        if any(skip in str(py_file) for skip in ['.venv', '__pycache__', 'build', 'dist']):
            continue
        
        try:
            content = py_file.read_text()
            relative_path = py_file.relative_to(project_root)
            
            # Find import statements
            for match in re.finditer(r'^import (\w+)', content, re.MULTILINE):
                imports[match.group(1)].append(str(relative_path))
            
            for match in re.finditer(r'^from (\w+)', content, re.MULTILINE):
                imports[match.group(1)].append(str(relative_path))
                
        except Exception as e:
            print(f"⚠️  Error reading {py_file}: {e}", file=sys.stderr)
    
    return imports


def categorize_imports(imports, project_root):
    """Categorize imports into stdlib, third-party, and local"""
    
    stdlib_imports = {}
    third_party = {}
    local = {}
    
    # Detect local modules (files/folders in project root)
    local_modules = set()
    for item in project_root.iterdir():
        if item.is_file() and item.suffix == '.py':
            local_modules.add(item.stem)
        elif item.is_dir() and not item.name.startswith('.'):
            local_modules.add(item.name)
    
    for module, files in imports.items():
        if module in STDLIB:
            stdlib_imports[module] = files
        elif module in local_modules or module.startswith('proj') or module in ['pySMC']:
            local[module] = files
        else:
            third_party[module] = files
    
    return stdlib_imports, third_party, local


def get_package_name(import_name):
    """Convert import name to package name"""
    return PACKAGE_MAP.get(import_name, import_name)


def main():
    project_root = Path(__file__).parent
    
    print("=" * 70)
    print("Project Dependency Analysis")
    print("=" * 70)
    print(f"Analyzing: {project_root}")
    print()
    
    # Find all imports
    imports = find_imports(project_root)
    
    # Categorize
    stdlib_imports, third_party, local = categorize_imports(imports, project_root)
    
    # Display results
    print("📦 THIRD-PARTY PACKAGES (Add these to pyproject.toml)")
    print("-" * 70)
    
    if third_party:
        for module in sorted(third_party.keys()):
            package_name = get_package_name(module)
            files = third_party[module]
            
            print(f"  {module:20s} → {package_name}")
            if len(files) <= 3:
                for f in files:
                    print(f"    • {f}")
            else:
                print(f"    • Used in {len(files)} files: {files[0]}, ...")
            print()
    else:
        print("  (No third-party imports found)")
    
    print()
    print("📚 STANDARD LIBRARY (Already included with Python)")
    print("-" * 70)
    print(f"  {len(stdlib_imports)} modules: {', '.join(sorted(list(stdlib_imports.keys())[:10]))}...")
    
    print()
    print("🏠 LOCAL MODULES (Your project code)")
    print("-" * 70)
    if local:
        print(f"  {', '.join(sorted(local.keys()))}")
    else:
        print("  (None found)")
    
    print()
    print("=" * 70)
    print("📝 Suggested pyproject.toml dependencies:")
    print("=" * 70)
    print()
    print("dependencies = [")
    
    # Group suggestions
    data_science = []
    web = []
    ml = []
    security = []
    other = []
    
    for module in sorted(third_party.keys()):
        package = get_package_name(module)
        
        if package in ['numpy', 'pandas', 'scipy', 'matplotlib']:
            data_science.append(f'  "{package}",')
        elif package in ['streamlit', 'fastapi', 'flask', 'django']:
            web.append(f'  "{package}",')
        elif package in ['scikit-learn', 'jax', 'optax', 'torch', 'tensorflow']:
            ml.append(f'  "{package}",')
        elif package in ['cryptography', 'pyotp', 'qrcode', 'python-dotenv']:
            security.append(f'  "{package}",')
        else:
            other.append(f'  "{package}",')
    
    if data_science:
        print("  # Data science")
        for line in data_science:
            print(line)
        print()
    
    if ml:
        print("  # Machine learning")
        for line in ml:
            print(line)
        print()
    
    if web:
        print("  # Web framework")
        for line in web:
            print(line)
        print()
    
    if security:
        print("  # Security")
        for line in security:
            print(line)
        print()
    
    if other:
        print("  # Other")
        for line in other:
            print(line)
    
    print("]")
    print()
    
    print("💡 Tip: Add version constraints like: 'numpy>=1.26'")
    print("💡 Check current versions with: pip show <package-name>")
    print()


if __name__ == '__main__':
    main()

