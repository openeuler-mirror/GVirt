#!/usr/bin/env python3
"""
Python Static Checker for xlite
Basic Python linting using flake8 and mypy
"""

import os
import sys
import subprocess
from pathlib import Path

class PythonChecker:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()
        self.xlite_dir = self.root_dir / "xlite"
        self.errors = []
        self.warnings = []

    def find_python_files(self):
        """Find all Python files"""
        python_files = []
        python_files.extend(self.xlite_dir.rglob('*.py'))
        python_files.extend(self.root_dir.rglob('*.py'))
        
        # Exclude build directories and cache files
        python_files = [f for f in python_files 
                       if 'build' not in str(f) 
                       and '__pycache__' not in str(f)
                       and '.egg-info' not in str(f)]
        
        return sorted(set(python_files))

    def check_flake8(self, file_path):
        """Run flake8 for basic linting"""
        try:
            result = subprocess.run(
                ['flake8', str(file_path), '--max-line-length=100', 
                 '--ignore=E501,W503,E203'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.warnings.append(f"{file_path}: {line}")
                return False
            return True
        except FileNotFoundError:
            self.warnings.append("flake8 not found, skipping flake8 check")
            return True
        except Exception as e:
            self.errors.append(f"{file_path}: Error running flake8: {e}")
            return False

    def check_mypy(self, file_path):
        """Run mypy for type checking"""
        try:
            result = subprocess.run(
                ['mypy', str(file_path), '--ignore-missing-imports', 
                 '--no-error-summary', '--no-pretty'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if line.strip() and 'Success' not in line:
                        self.warnings.append(f"{file_path}: {line}")
                return False
            return True
        except FileNotFoundError:
            self.warnings.append("mypy not found, skipping type check")
            return True
        except Exception as e:
            self.errors.append(f"{file_path}: Error running mypy: {e}")
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

    def run_checks(self, check_flake8_flag=True, check_mypy_flag=True):
        """Run all checks"""
        python_files = self.find_python_files()
        
        if not python_files:
            print("No Python files found")
            return True
        
        print(f"Checking {len(python_files)} Python files...")
        
        all_passed = True
        
        for file_path in python_files:
            print(f"  Checking {file_path.relative_to(self.root_dir)}", end='\r')
            
            self.check_syntax(file_path)
            
            if check_flake8_flag:
                self.check_flake8(file_path)
            
            if check_mypy_flag:
                self.check_mypy(file_path)
        
        print()
        
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
    
    args = parser.parse_args()
    
    checker = PythonChecker(args.root_dir)
    result = checker.run_checks(
        check_flake8_flag=not args.no_flake8,
        check_mypy_flag=not args.no_mypy
    )
    
    sys.exit(0 if result else 1)

if __name__ == '__main__':
    main()
