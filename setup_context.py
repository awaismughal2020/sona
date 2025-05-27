#!/usr/bin/env python3
"""
Quick setup script to add ContextType to constants.py if missing
Run this before starting the backend
"""

import os
import sys


def add_context_type_to_constants():
    """Add ContextType enum to constants.py if it doesn't exist"""
    constants_file = "utils/constants.py"

    if not os.path.exists(constants_file):
        print(f"❌ {constants_file} not found!")
        return False

    # Read current content
    with open(constants_file, 'r') as f:
        content = f.read()

    # Check if ContextType already exists
    if "class ContextType" in content:
        print("✅ ContextType already exists in constants.py")
        return True

    # Add ContextType enum before SONA_PERSONA
    context_type_code = '''

class ContextType(Enum):
    """Types of context information."""
    USER_PREFERENCE = "user_preference"
    TOPIC = "topic"
    ENTITY = "entity"
    TASK = "task"
    SEARCH_HISTORY = "search_history"
    IMAGE_HISTORY = "image_history"
    CONVERSATION_FLOW = "conversation_flow"

'''

    # Find where to insert (before SONA_PERSONA or at the end of enums)
    if "# SONA Persona Configuration" in content:
        content = content.replace(
            "# SONA Persona Configuration",
            context_type_code + "# SONA Persona Configuration"
        )
    elif "SONA_PERSONA = {" in content:
        content = content.replace(
            "SONA_PERSONA = {",
            context_type_code + "SONA_PERSONA = {"
        )
    else:
        # Add at the end of the file
        content += context_type_code

    # Write back to file
    with open(constants_file, 'w') as f:
        f.write(content)

    print("✅ Added ContextType to constants.py")
    return True


def create_contexts_directory():
    """Create contexts directory if it doesn't exist"""
    contexts_dir = "contexts"

    if not os.path.exists(contexts_dir):
        os.makedirs(contexts_dir)
        print(f"✅ Created {contexts_dir} directory")
    else:
        print(f"✅ {contexts_dir} directory already exists")

    return True


def check_file_structure():
    """Check if required files exist"""
    required_files = [
        "utils/conversation_context.py",
        "utils/context_store.py",
        "utils/constants.py",
        "ai/orchestrator.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ All required files exist")
    return True


def main():
    """Main setup function"""
    print("🔧 Setting up SONA Context Management...")

    # Check file structure
    if not check_file_structure():
        print("\n❌ Setup failed! Please ensure all required files are in place.")
        print("\nRequired files:")
        print("- utils/conversation_context.py")
        print("- utils/context_store.py")
        print("- utils/constants.py")
        print("- ai/orchestrator.py (should be the enhanced version)")
        return False

    # Add ContextType to constants
    if not add_context_type_to_constants():
        print("❌ Failed to add ContextType to constants.py")
        return False

    # Create contexts directory
    if not create_contexts_directory():
        print("❌ Failed to create contexts directory")
        return False

    print("\n✅ Context management setup completed successfully!")
    print("\nYou can now start your SONA backend with:")
    print("docker-compose up --build")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
