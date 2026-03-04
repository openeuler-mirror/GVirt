#!/usr/bin/env python3
"""
Python Static Checker for xlite
Basic Python linting using flake8 and mypy
"""

import os
import sys
import subprocess
import hashlib
import json
from pathlib import Path

class PythonChecker:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()
        self.xlite_dir = self.root_dir / "xlite"
        self.errors = []
        self.warnings = []
        self.cache_dir = self.root_dir / ".static_check_cache"
        self.cache_file = self.cache_dir / "python_check_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk"""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_cache(self):
        """Save cache to disk"""
        self.cache_dir.mkdir(exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def _get_file_hash(self, file_path):
        """Get hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None

    def _is_file_changed(self, file_path):
        """Check if file has changed since last check"""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return True

        file_str = str(file_path)
        if file_str in self.cache:
            return self.cache[file_str] != file_hash

        return True

    def _update_cache(self, file_path):
        """Update cache for a file"""
        file_hash = self._get_file_hash(file_path)
        if file_hash:
            self.cache[str(file_path)] = file_hash

    def find_python_files(self, use_cache=True):
        """Find all Python files"""
        python_files = []

        # Only search in xlite_dir (root_dir contains xlite_dir, so no need to search both)
        python_files.extend(self.xlite_dir.rglob('*.py'))
        
        # Exclude build directories and cache files
        exclude_patterns = ['build', '__pycache__', '.egg-info', '.pytest_cache', '.static_check_cache']
        python_files = [f for f in python_files 
                       if not any(pattern in str(f) for pattern in exclude_patterns)]

        python_files = sorted(set(python_files))

        # Filter out unchanged files if cache is enabled
        if use_cache:
            changed_files = [f for f in python_files if self._is_file_changed(f)]
            if len(changed_files) < len(python_files):
                print(f"  Cache hit: {len(python_files) - len(changed_files)} files unchanged, checking {len(changed_files)} files")
            return changed_files
        
        return python_files

    def check_flake8_batch(self, file_paths):
        """Run flake8 for basic linting on all files at once"""
        if not file_paths:
            return True

        try:
            file_list = [str(f) for f in file_paths]
            result = subprocess.run(
                ['flake8'] + file_list + ['--max-line-length=100',
                 '--ignore=E501,W503,E203'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.warnings.append(line)
                return False
            return True
        except FileNotFoundError:
            self.warnings.append("flake8 not found, skipping flake8 check")
            return True
        except Exception as e:
            self.errors.append(f"Error running flake8: {e}")
            return False

    def check_mypy_batch(self, file_paths):
        """Run mypy for type checking on all files at once"""
        if not file_paths:
            return True

        try:
            file_list = [str(f) for f in file_paths]
            result = subprocess.run(
                ['mypy'] + file_list + ['--ignore-missing-imports',
                 '--no-error-summary', '--no-pretty', '--show-error-codes'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if line.strip() and 'Success' not in line:
                        self.warnings.append(line)
                return False
            return True
        except FileNotFoundError:
            self.warnings.append("mypy not found, skipping type check")
            return True
        except Exception as e:
            self.errors.append(f"Error running mypy: {e}")
            return False

    def check_syntax(self, file_path):
        """Check Python syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(file_path), 'exec')
            return True
        except SyntaxError as e:
            self.errors.append(f"{file_path}: Syntax error at line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            self.errors.append(f"{file_path}: Error reading file: {e}")
            return False

    def run_checks(self, check_flake8_flag=True, check_mypy_flag=True, use_cache=False):
        """Run all checks"""
        python_files = self.find_python_files(use_cache=use_cache)
        
        if not python_files:
            print("No Python files found")
            return True
        
        print(f"Checking {len(python_files)} Python files...")
        
        all_passed = True
        
        # Check syntax for all files
        for file_path in python_files:
            print(f"  Checking {file_path.relative_to(self.root_dir)}", end='\r')
            if not self.check_syntax(file_path):
                all_passed = False

        print()

        # Run batch checks
        if (check_flake8_flag or check_mypy_flag) and python_files:
            if check_flake8_flag:
                print("Running flake8...")
                self.check_flake8_batch(python_files)
            
            if check_mypy_flag:
                print("Running mypy...")
                self.check_mypy_batch(python_files)
        
        # Update cache for checked files
        if use_cache:
            for file_path in python_files:
                self._update_cache(file_path)
            self._save_cache()
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  {error}")
            all_passed = False
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings[:20]:  # Limit warnings
                print(f"  {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")
        
        if all_passed and not self.warnings:
            print("\n✅ All Python checks passed!")
        
        return all_passed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Python Static Checker for xlite')
    parser.add_argument('--root-dir', default=str(Path(__file__).parent.parent),
                       help='Root directory of xlite project')
    parser.add_argument('--no-flake8', action='store_true',
                       help='Skip flake8 check')
    parser.add_argument('--no-mypy', action='store_true',
                       help='Skip mypy check')
    parser.add_argument('--use-cache', action='store_true',
                       help='Enable caching to skip unchanged files')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear the cache before running checks')
    
    args = parser.parse_args()
    
    checker = PythonChecker(args.root_dir)

    if args.clear_cache:
        if checker.cache_file.exists():
            checker.cache_file.unlink()
            print("Cache cleared")
        checker.cache = {}

    result = checker.run_checks(
        check_flake8_flag=not args.no_flake8,
        check_mypy_flag=not args.no_mypy,
        use_cache=args.use_cache
    )
    
    sys.exit(0 if result else 1)

if __name__ == '__main__':
    main()
