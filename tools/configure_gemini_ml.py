"""
Script to configure ML Engineer to use Gemini 3 Flash Preview.

This script updates the .env file to set:
- ML_ENGINEER_PROVIDER=google
- ML_ENGINEER_MODEL=gemini-3-flash-preview (optional, already default)

Run this script to switch from OpenRouter (z-ai/glm-4.7) to Google Gemini.
"""

import os
from pathlib import Path

def update_env_for_gemini():
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        return False
    
    # Read current .env
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if ML_ENGINEER_PROVIDER exists
    provider_found = False
    model_found = False
    updated_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Update ML_ENGINEER_PROVIDER
        if stripped.startswith('ML_ENGINEER_PROVIDER='):
            updated_lines.append('ML_ENGINEER_PROVIDER=google\n')
            provider_found = True
            print(f"✅ Updated: ML_ENGINEER_PROVIDER=google")
        
        # Update ML_ENGINEER_MODEL (optional)
        elif stripped.startswith('ML_ENGINEER_MODEL='):
            updated_lines.append('ML_ENGINEER_MODEL=gemini-3-flash-preview\n')
            model_found = True
            print(f"✅ Updated: ML_ENGINEER_MODEL=gemini-3-flash-preview")
        
        # Keep other lines unchanged
        else:
            updated_lines.append(line)
    
    # Add missing variables
    if not provider_found:
        updated_lines.append('\n# ML Engineer LLM Configuration\n')
        updated_lines.append('ML_ENGINEER_PROVIDER=google\n')
        print(f"✅ Added: ML_ENGINEER_PROVIDER=google")
    
    if not model_found:
        updated_lines.append('ML_ENGINEER_MODEL=gemini-3-flash-preview\n')
        print(f"✅ Added: ML_ENGINEER_MODEL=gemini-3-flash-preview")
    
    # Verify GOOGLE_API_KEY exists
    has_google_key = any(line.strip().startswith('GOOGLE_API_KEY=') for line in updated_lines)
    if not has_google_key:
        print(f"\n⚠️ WARNING: GOOGLE_API_KEY not found in .env")
        print(f"   Please add: GOOGLE_API_KEY=your_key_here")
        return False
    
    # Write updated .env
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"\n✅ SUCCESS: .env updated for Gemini 3 Flash Preview")
    print(f"\n📋 Current ML Engineer Configuration:")
    print(f"   Provider: google")
    print(f"   Model: gemini-3-flash-preview")
    print(f"\n🚀 Next: Run app.py to test with new model")
    
    return True

if __name__ == '__main__':
    success = update_env_for_gemini()
    exit(0 if success else 1)
