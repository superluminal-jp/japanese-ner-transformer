#!/usr/bin/env python3
"""
Test runner script for Japanese NER Transformer.

This script provides a convenient way to run different test suites
and generate reports.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install -r requirements-dev.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test runner for Japanese NER Transformer")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests only')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', '-x', action='store_true', help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ['pytest']
    
    if args.verbose:
        base_cmd.append('-v')
    
    if args.fail_fast:
        base_cmd.append('-x')
    
    # Determine test scope
    test_commands = []
    
    if args.unit:
        cmd = base_cmd + ['tests/unit/', '-m', 'unit or not (integration or e2e)']
        test_commands.append((cmd, "Unit Tests"))
    
    elif args.integration:
        cmd = base_cmd + ['tests/integration/', '-m', 'integration']
        test_commands.append((cmd, "Integration Tests"))
    
    elif args.e2e:
        cmd = base_cmd + ['tests/e2e/', '-m', 'e2e']
        test_commands.append((cmd, "End-to-End Tests"))
    
    elif args.coverage:
        cmd = base_cmd + [
            'tests/',
            '--cov=src',
            '--cov-report=html',
            '--cov-report=term-missing',
            '--cov-report=xml'
        ]
        test_commands.append((cmd, "All Tests with Coverage"))
    
    else:
        # Run all tests by default
        cmd = base_cmd + ['tests/']
        test_commands.append((cmd, "All Tests"))
    
    # Run the commands
    success_count = 0
    total_count = len(test_commands)
    
    for cmd, description in test_commands:
        if run_command(cmd, description):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()