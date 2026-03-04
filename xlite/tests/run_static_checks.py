#!/usr/bin/env python3
"""
Unified Static Checker Entry Point for xlite
Runs C++ and Python static checks
"""

import os
import sys
import subprocess
from pathlib import Path

def run_cpp_checks(root_dir, args):
    """Run C++ static checks"""
    print("\n" + "="*60)
    print("Running C++ Static Checks")
    print("="*60 + "\n")
    
    check_script = Path(__file__).parent / "cpp_check.py"
    
    cmd = [sys.executable, str(check_script), '--root-dir', root_dir]
    if args.no_cpp_format:
        cmd.append('--no-format')
    if args.no_cpp_copyright:
        cmd.append('--no-copyright')
    if args.no_cpp_header_guard:
        cmd.append('--no-header-guard')
    if args.no_cpp_tidy:
        cmd.append('--no-tidy')
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_python_checks(root_dir, args):
    """Run Python static checks"""
    print("\n" + "="*60)
    print("Running Python Static Checks")
    print("="*60 + "\n")
    
    check_script = Path(__file__).parent / "python_check.py"
    
    cmd = [sys.executable, str(check_script), '--root-dir', root_dir]
    if args.no_python_flake8:
        cmd.append('--no-flake8')
    if args.no_python_mypy:
        cmd.append('--no-mypy')
    if args.use_cache:
        cmd.append('--use-cache')
    if args.clear_cache:
        cmd.append('--clear-cache')
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified Static Checker for xlite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all checks
  python run_static_checks.py
  
  # Run only C++ checks
  python run_static_checks.py --cpp-only
  
  # Run only Python checks
  python run_static_checks.py --python-only
  
  # Skip specific C++ checks
  python run_static_checks.py --no-cpp-tidy --no-cpp-format
  
  # Skip specific Python checks
  python run_static_checks.py --no-python-mypy
        """
    )
    
    parser.add_argument('--root-dir', default=str(Path(__file__).parent.parent),
                       help='Root directory of xlite project')
    parser.add_argument('--cpp-only', action='store_true',
                       help='Run only C++ checks')
    parser.add_argument('--python-only', action='store_true',
                       help='Run only Python checks')
    
    # C++ check options
    parser.add_argument('--no-cpp-format', action='store_true',
                       help='Skip C++ code format check')
    parser.add_argument('--no-cpp-copyright', action='store_true',
                       help='Skip C++ copyright check')
    parser.add_argument('--no-cpp-header-guard', action='store_true',
                       help='Skip C++ header guard check')
    parser.add_argument('--no-cpp-tidy', action='store_true',
                       help='Skip clang-tidy check')
    
    # Python check options
    parser.add_argument('--no-python-flake8', action='store_true',
                       help='Skip Python flake8 check')
    parser.add_argument('--no-python-mypy', action='store_true',
                       help='Skip Python mypy check')
    parser.add_argument('--use-cache', action='store_true',
                       help='Enable caching for Python checks')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before running Python checks')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("xlite Static Checker")
    print("="*60)
    
    results = {}
    
    if not args.python_only:
        results['cpp'] = run_cpp_checks(args.root_dir, args)
    
    if not args.cpp_only:
        results['python'] = run_python_checks(args.root_dir, args)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for check_type, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {check_type.upper():10s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
