#!/usr/bin/env python3
"""
C++ Static Checker for xlite
Checks code formatting, copyright, header guards, and runs clang-tidy
"""

import sys
import subprocess
import re
import shutil
from pathlib import Path


class CppChecker:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()
        self.csrc_dir = self.root_dir / "csrc"
        self.errors = []
        self.warnings = []

    def check_copyright(self, file_path):
        """Check if file has proper copyright header"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            copyright_pattern = r'Copyright.*Huawei Technologies'
            has_copyright = any(re.search(copyright_pattern, line) for line in first_lines)
            
            if not has_copyright:
                self.errors.append(f"{file_path}: Missing copyright header")
                return False
            return True
        except Exception as e:
            self.errors.append(f"{file_path}: Error reading file: {e}")
            return False

    def check_header_guard(self, file_path):
        """Check if header file has proper header guard"""
        if not file_path.suffix == '.h':
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # Check for #pragma once or #ifndef
            has_pragma_once = False
            has_ifndef = False
            ifndef_match = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('#pragma once'):
                    has_pragma_once = True
                    break
                elif stripped.startswith('#ifndef'):
                    has_ifndef = True
                    match = re.match(r'#ifndef\s+(\S+)', line)
                    if match:
                        ifndef_match = match.group(1)
                    break

            if not has_pragma_once and not has_ifndef:
                self.errors.append(f"{file_path}: Missing header guard")
                return False

            # If using #ifndef, check for matching #define
            if has_ifndef and ifndef_match:
                has_define = any(re.match(rf'#define\s+{re.escape(ifndef_match)}', line)
                                for line in lines)
                if not has_define:
                    self.errors.append(f"{file_path}: Missing #define for header guard {ifndef_match}")
                    return False

            return True
        except Exception as e:
            self.errors.append(f"{file_path}: Error checking header guard: {e}")
            return False

    def check_format(self, file_path):
        """Check code formatting with clang-format"""
        try:
            result = subprocess.run(
                ['clang-format', '--dry-run', '--Werror', str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.errors.append(f"{file_path}: Code formatting issues found")
                return False
            return True
        except FileNotFoundError:
            self.errors.append("clang-format not found, skipping format check")
            return True
        except Exception as e:
            self.errors.append(f"{file_path}: Error running clang-format: {e}")
            return False

    def build_project(self):
        """Build project to generate operator header files"""
        try:
            build_dir = self.root_dir / 'build'
            if build_dir.exists():
                shutil.rmtree(build_dir)

            build_dir.mkdir(parents=True, exist_ok=True)

            cmake_configure = subprocess.run(
                ['cmake', '-B', 'build'],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if cmake_configure.returncode != 0:
                self.errors.append(f"Failed to configure cmake: {cmake_configure.stderr}")
                return False

            cmake_build = subprocess.run(
                ['cmake', '--build', 'build', '-j'],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if cmake_build.returncode != 0:
                self.errors.append(f"Failed to build project: {cmake_build.stderr}")
                return False

            cmake_install = subprocess.run(
                ['cmake', '--install', 'build'],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if cmake_install.returncode != 0:
                self.errors.append(f"Failed to install project: {cmake_install.stderr}")
                return False

            return True
        except FileNotFoundError:
            self.errors.append("cmake not found, skipping project build")
            return False
        except Exception as e:
            self.errors.append(f"Error building project: {e}")
            return False

    def generate_compile_commands(self):
        """Generate compile_commands.json using cmake"""
        try:
            build_dir = self.root_dir / 'build'

            result = subprocess.run(
                ['cmake', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '..'],
                capture_output=True,
                text=True,
                cwd=build_dir
            )

            if result.returncode != 0:
                self.errors.append(f"Failed to generate compile_commands.json: {result.stderr}")
                return False
            
            compile_commands = build_dir / 'compile_commands.json'
            if not compile_commands.exists():
                self.errors.append("compile_commands.json not found after cmake")
                return False

            return True
        except FileNotFoundError:
            self.errors.append("cmake not found, skipping compile_commands.json generation")
            return False
        except Exception as e:
            self.errors.append(f"Error generating compile_commands.json: {e}")
            return False

    def run_clang_tidy(self, file_path):
        """Run clang-tidy static analysis"""
        try:
            config_file = '.clang-tidy'
            if 'kernels' in str(file_path) or 'csrc/ascend.h' in str(file_path):
                return True

            build_dir = self.root_dir / 'build'
            result = subprocess.run(
                ['clang-tidy', str(file_path), f'--config-file={config_file}',
                  f'-p={build_dir}'],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode != 0 or result.stdout.strip():
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.errors.append(f"{file_path}: {line}")
                return False
            return True
        except FileNotFoundError:
            self.errors.append("clang-tidy not found, skipping static analysis")
            return True
        except Exception as e:
            self.errors.append(f"{file_path}: Error running clang-tidy: {e}")
            return False

    def find_cpp_files(self):
        """Find all C++ source and header files"""
        cpp_files = []
        for ext in ['*.cpp', '*.h', '*.hpp']:
            cpp_files.extend(self.csrc_dir.rglob(ext))
        return sorted(cpp_files)

    def run_checks(self, check_format=True, check_copyright_flag=True, 
                   check_header_guard_flag=True, run_tidy=True):
        """Run all checks"""
        cpp_files = self.find_cpp_files()
        
        if not cpp_files:
            print("No C++ files found")
            return True
        
        print(f"Checking {len(cpp_files)} C++ files...")
        
        all_passed = True
        
        if run_tidy:
            print("Building project to generate operator header files...")
            if not self.build_project():
                all_passed = False

            print("Generating compile_commands.json...")
            if not self.generate_compile_commands():
                all_passed = False
        
        for file_path in cpp_files:
            print(f"\033[K  Checking {file_path.relative_to(self.root_dir)}", end='\r')
            
            if check_copyright_flag:
                self.check_copyright(file_path)
            
            if check_header_guard_flag:
                self.check_header_guard(file_path)
            
            if check_format:
                self.check_format(file_path)
            
            if run_tidy:
                self.run_clang_tidy(file_path)
        
        print()
        
        if self.errors:
            print(f"\n❌  {len(self.errors)} ERRORS:")
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
            print("\n✅ All C++ checks passed!")
        
        return all_passed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='C++ Static Checker for xlite')
    parser.add_argument('--root-dir', default=str(Path(__file__).parent.parent),
                       help='Root directory of xlite project')
    parser.add_argument('--no-format', action='store_true',
                       help='Skip code format check')
    parser.add_argument('--no-copyright', action='store_true',
                       help='Skip copyright check')
    parser.add_argument('--no-header-guard', action='store_true',
                       help='Skip header guard check')
    parser.add_argument('--no-tidy', action='store_true',
                       help='Skip clang-tidy check')
    
    args = parser.parse_args()
    
    checker = CppChecker(args.root_dir)
    result = checker.run_checks(
        check_format=not args.no_format,
        check_copyright_flag=not args.no_copyright,
        check_header_guard_flag=not args.no_header_guard,
        run_tidy=not args.no_tidy
    )
    
    sys.exit(0 if result else 1)

if __name__ == '__main__':
    main()
