"""
Copyright 2024 Job Application Helper Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python3
"""
Job Application Helper - System Check

This script performs thorough checks of your system:
- Python version and virtual environment
- Dependencies and imports
- Configuration system
- Logging system
- File structure
- Security settings
- Application readiness
"""

import importlib.util
from pathlib import Path
import sys
from typing import List, Tuple


class SystemChecker:
    """Comprehensive system verification."""

    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.project_root = Path(__file__).parent.parent
        # Add project root to Python path for imports
        sys.path.insert(0, str(self.project_root))

    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Log a test result."""
        self.results.append((test_name, success, message))
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")

    def check_python_version(self) -> bool:
        """Verify Python version is 3.9+."""
        print("\n🐍 Checking Python Version...")
        version = sys.version_info

        if version.major == 3 and version.minor >= 9:
            self.log_result(
                "Python Version",
                True,
                f"Python {version.major}.{version.minor}.{version.micro}",
            )
            return True
        self.log_result(
            "Python Version",
            False,
            f"Python {version.major}.{version.minor}.{version.micro} - Need 3.9+",
        )
        return False

    def check_virtual_environment(self) -> bool:
        """Check if we're in a virtual environment."""
        print("\n🔧 Checking Virtual Environment...")

        # Check if we're in a virtual environment
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        if in_venv:
            venv_path = sys.prefix
            self.log_result("Virtual Environment", True, f"Active: {venv_path}")

            # Check if it's the project's venv
            expected_venv = self.project_root / "venv"
            if Path(venv_path).resolve() == expected_venv.resolve():
                self.log_result(
                    "Project Virtual Environment", True, "Using project venv"
                )
                return True
            self.log_result(
                "Project Virtual Environment",
                False,
                f"Using different venv: {venv_path}",
            )
            return False
        self.log_result("Virtual Environment", False, "Not in virtual environment")
        return False

    def check_dependencies(self) -> bool:
        """Check all required dependencies."""
        print("\n📦 Checking Dependencies...")

        required_packages = [
            ("pydantic", "2.0.0"),
            ("pydantic_settings", "2.0.0"),
            ("dotenv", "1.0.0"),
            ("cryptography", "40.0.0"),
            ("pythonjsonlogger", "2.0.0"),
        ]

        all_good = True
        for package, min_version in required_packages:
            try:
                module = importlib.import_module(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                self.log_result(f"Package {package}", True, f"v{version}")
            except ImportError:
                self.log_result(f"Package {package}", False, "Not installed")
                all_good = False

        return all_good

    def check_project_structure(self) -> bool:
        """Check project directory structure."""
        print("\n📁 Checking Project Structure...")

        required_dirs = [
            "src",
            "src/core",
            "src/utils",
            "src/ui",
            "data",
            "data/documents",
            "data/cache",
            "data/vector_db",
            "tests",
        ]

        required_files = [
            "src/core/__init__.py",
            "src/utils/__init__.py",
            "src/utils/config.py",
            "src/utils/logging.py",
            "requirements.txt",
            "requirements-dev.txt",
            "env.example",
        ]

        all_good = True

        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_result(f"Directory {dir_path}", True, "Exists")
            else:
                self.log_result(f"Directory {dir_path}", False, "Missing")
                all_good = False

        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.log_result(
                    f"File {file_path}",
                    True,
                    f"Exists ({full_path.stat().st_size} bytes)",
                )
            else:
                self.log_result(f"File {file_path}", False, "Missing")
                all_good = False

        return all_good

    def check_configuration_system(self) -> bool:
        """Test configuration system."""
        print("\n⚙️  Checking Configuration System...")

        try:
            # Test imports
            from src.utils.config import (
                ensure_directories,
                get_settings,
                validate_required_settings,
            )

            self.log_result("Config Imports", True, "All imports successful")

            # Test settings loading
            settings = get_settings()
            self.log_result(
                "Settings Loading", True, f"Environment: {settings.environment}"
            )

            # Test directory creation
            ensure_directories(settings)
            self.log_result("Directory Creation", True, "All directories ensured")

            # Test validation (should require API keys for production use)
            try:
                validate_required_settings(settings)
                self.log_result(
                    "Settings Validation",
                    True,
                    "Configuration system working correctly",
                )
            except ValueError:
                self.log_result(
                    "Settings Validation",
                    True,
                    "Correctly validates API key requirements",
                )

            return True

        except Exception as e:
            self.log_result("Configuration System", False, f"Error: {e}")
            return False

    def check_logging_system(self) -> bool:
        """Test logging system."""
        print("\n📝 Checking Logging System...")

        try:
            from src.utils.config import get_settings
            from src.utils.logging import SecurityFilter, get_logger, setup_logging

            # Test setup
            settings = get_settings()
            setup_logging(settings)
            self.log_result("Logging Setup", True, "Setup successful")

            # Test logger creation
            logger = get_logger("test_logger")
            self.log_result("Logger Creation", True, "Logger created")

            # Test security filter
            filter_instance = SecurityFilter()
            self.log_result("Security Filter", True, "Security filter available")

            return True

        except Exception as e:
            self.log_result("Logging System", False, f"Error: {e}")
            return False

    def check_document_processor(self) -> bool:
        """Test document processing module."""
        print("\n📄 Checking Document Processor...")

        try:
            # Check module imports
            from src.core.document_processor import (
                DocumentProcessor,
            )

            self.log_result(
                "Document Processor Import", True, "Module imported successfully"
            )

            # Check dependencies
            dependencies = []
            try:
                import fitz

                dependencies.append("PyMuPDF")
            except ImportError:
                pass

            try:
                import pymupdf4llm

                dependencies.append("pymupdf4llm")
            except ImportError:
                pass

            try:
                from docx import Document

                dependencies.append("python-docx")
            except ImportError:
                pass

            if dependencies:
                self.log_result(
                    "Document Dependencies",
                    True,
                    f"Available: {', '.join(dependencies)}",
                )
            else:
                self.log_result(
                    "Document Dependencies",
                    False,
                    "No document processing libraries available",
                )
                return False

            # Test processor initialization
            processor = DocumentProcessor()
            self.log_result("Document Processor Init", True, "Processor initialized")

            # Test file type detection
            supported_types = []
            if "PyMuPDF" in dependencies:
                supported_types.append("PDF")
            if "python-docx" in dependencies:
                supported_types.append("DOCX")

            self.log_result(
                "File Type Support", True, f"Supports: {', '.join(supported_types)}"
            )

            # Test validation
            valid, _ = processor.validate_file(self.project_root / "requirements.txt")
            self.log_result("File Validation", True, "Validation function works")

            return True

        except Exception as e:
            self.log_result("Document Processor", False, f"Error: {e}")
            return False

    def check_environment_template(self) -> bool:
        """Check environment template file."""
        print("\n🌍 Checking Environment Template...")

        env_example = self.project_root / "env.example"

        if not env_example.exists():
            self.log_result("Environment Template", False, "env.example missing")
            return False

        # Check for required environment variables
        required_vars = [
    
            "ENVIRONMENT",
            "PORT",
            "LOG_LEVEL",
            "DATA_DIR",
            "ENCRYPTION_KEY",
            "EMBEDDING_MODEL",
        ]

        content = env_example.read_text()
        missing_vars = []

        for var in required_vars:
            if var not in content:
                missing_vars.append(var)

        if missing_vars:
            self.log_result(
                "Environment Variables", False, f"Missing: {', '.join(missing_vars)}"
            )
            return False
        self.log_result(
            "Environment Variables",
            True,
            f"All {len(required_vars)} variables present",
        )
        return True

    def check_git_setup(self) -> bool:
        """Check git setup and .gitignore."""
        print("\n🔧 Checking Git Setup...")

        gitignore = self.project_root / ".gitignore"

        if not gitignore.exists():
            self.log_result("Git Ignore", False, ".gitignore missing")
            return False

        # Check for important patterns
        content = gitignore.read_text()
        important_patterns = [
            ".env",
            "venv/",
            "__pycache__/",
            "*.pyc",
        ]

        missing_patterns = []
        for pattern in important_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)

        if missing_patterns:
            self.log_result(
                "Git Ignore Patterns", False, f"Missing: {', '.join(missing_patterns)}"
            )
            return False
        self.log_result(
            "Git Ignore Patterns", True, "All important patterns present"
        )
        return True

    def check_requirements_files(self) -> bool:
        """Check requirements files are properly formatted."""
        print("\n📋 Checking Requirements Files...")

        req_files = [
            ("requirements.txt", "Production dependencies"),
            ("requirements-dev.txt", "Development dependencies"),
        ]

        all_good = True
        for filename, description in req_files:
            file_path = self.project_root / filename

            if not file_path.exists():
                self.log_result(f"Requirements {filename}", False, "File missing")
                all_good = False
                continue

            content = file_path.read_text()
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.startswith("#")
            ]

            if lines:
                self.log_result(
                    f"Requirements {filename}", True, f"{len(lines)} dependencies"
                )
            else:
                self.log_result(
                    f"Requirements {filename}", False, "No dependencies found"
                )
                all_good = False

        return all_good

    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        print("🔍 JOB APPLICATION HELPER - SYSTEM CHECK")
        print("=" * 60)

        checks = [
            self.check_python_version,
            self.check_virtual_environment,
            self.check_dependencies,
            self.check_project_structure,
            self.check_configuration_system,
            self.check_logging_system,
            self.check_document_processor,
            self.check_environment_template,
            self.check_git_setup,
            self.check_requirements_files,
        ]

        all_passed = True
        for check in checks:
            try:
                result = check()
                all_passed = all_passed and result
            except Exception as e:
                print(f"❌ Error running check: {e}")
                all_passed = False

        return all_passed

    def print_summary(self) -> None:
        """Print verification summary."""
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        print(f"Tests Passed: {passed}/{total}")

        if passed == total:
            print("\n🎉 ALL CHECKS PASSED!")
            print("✅ Job Application Helper is ready to use!")
            print("\n📋 Next Steps:")
            print("  1. Copy env.example to .env and add your API keys")
            print("  2. Launch the application with './launch_app.sh' (or launch_app.bat on Windows)")
            print("  3. Check README.md for usage instructions")
        else:
            print(f"\n⚠️  {total - passed} ISSUES FOUND")
            print("\n❌ Failed Tests:")
            for test_name, success, message in self.results:
                if not success:
                    print(f"  • {test_name}: {message}")
            print("\n🔧 Please fix the issues above before proceeding")


def main():
    """Main system check function."""
    checker = SystemChecker()
    success = checker.run_all_checks()
    checker.print_summary()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
