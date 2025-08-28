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
Job Application Helper - System Status

This script helps you check your system status:
- Checks virtual environment
- Verifies setup
- Provides helpful commands
- Shows system information
"""

import os
from pathlib import Path
import sys


def check_virtual_env():
    """Check if we're in the correct virtual environment."""
    project_root = Path(__file__).parent.parent
    expected_venv = project_root / "venv"

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("❌ Not in virtual environment!")
        print("\n🔧 To activate the virtual environment, run:")
        print("   source venv/bin/activate")
        print("   # On Windows: venv\\Scripts\\activate")
        return False

    # Check if it's the correct venv
    current_venv = Path(sys.prefix).resolve()
    if current_venv != expected_venv.resolve():
        print(f"⚠️  Using different virtual environment: {current_venv}")
        print(f"   Expected: {expected_venv}")
        print("\n🔧 To use the project virtual environment:")
        print("   deactivate")
        print("   source venv/bin/activate")
        return False

    print(f"✅ Using correct virtual environment: {current_venv}")
    return True


def show_python_info():
    """Show Python version and environment info."""
    print(f"\n🐍 Python: {sys.version}")
    print(f"📍 Location: {sys.executable}")
    print(f"🔧 Virtual Environment: {sys.prefix}")


def show_project_status():
    """Show current project status."""
    project_root = Path(__file__).parent.parent

    print("\n📋 PROJECT STATUS")
    print("=" * 40)

    # Check README for user guidance
    readme = project_root / "README.md"
    if readme.exists():
        print("✅ User documentation available (README.md)")
    else:
        print("❌ User documentation missing")

    # Check configuration
    env_example = project_root / "env.example"
    env_file = project_root / ".env"

    if env_example.exists():
        print("✅ Environment template available")
    else:
        print("❌ Environment template missing")

    if env_file.exists():
        print("✅ Environment configuration found")
    else:
        print("⚠️  Environment configuration missing (.env)")
        print("   Copy env.example to .env and add your API keys")

    # Check data directories
    data_dirs = ["data/documents", "data/cache"]
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path} directory ready")
        else:
            print(f"❌ {dir_path} directory missing")


def show_available_commands():
    """Show available commands."""
    print("\n🛠️  AVAILABLE COMMANDS")
    print("=" * 40)

    commands = [
        ("python scripts/check_system.py", "Run comprehensive system verification"),
        ("python scripts/status.py", "Show this system status"),
        ("python -m pytest tests/", "Run tests (when available)"),
        ("python -m black src/", "Format code with Black"),
        ("python -m flake8 src/", "Lint code with Flake8"),
        ("streamlit run src/ui/streamlit_app.py", "Start UI (when available)"),
        ("uvicorn src.api.main:app --reload", "Start API server (when available)"),
    ]

    for command, description in commands:
        print(f"  {command}")
        print(f"    └─ {description}")
        print()


def show_next_steps():
    """Show next development steps."""
    print("\n🎯 NEXT STEPS")
    print("=" * 40)

    steps = [
        "1. Ensure .env file is configured with your API keys",
        "2. Run 'python scripts/check_system.py' to verify setup",
        "3. Check README.md for usage instructions",
        "4. Launch the application with './launch_app.sh' (or launch_app.bat on Windows)",
    ]

    for step in steps:
        print(f"  {step}")


def main():
    """Main development interface."""

    print("🚀 JOB APPLICATION HELPER - SYSTEM STATUS")
    print("=" * 60)

    # Check virtual environment
    venv_ok = check_virtual_env()

    if not venv_ok:
        print("\n❌ Please activate the virtual environment first!")
        return False

    # Show Python info
    show_python_info()

    # Show project status
    show_project_status()

    # Show available commands
    show_available_commands()

    # Show next steps
    show_next_steps()

    print("\n📋 System Menu:")
    print("1. Show system status")
    print("2. Run verification script")
    print("0. Exit")

    choice = input("\nEnter your choice (0-2): ").strip()

    if choice == "1":
        show_python_info()
        show_project_status()
    elif choice == "2":
        print("\n🔍 Running system check...")
        os.system("python scripts/check_system.py")
    elif choice == "0":
        print("👋 Happy coding!")
        return None
    else:
        print("❌ Invalid choice. Please try again.")

    print("\n" + "=" * 60)
    print("Happy coding! 🎉")

    return True


if __name__ == "__main__":
    main()
