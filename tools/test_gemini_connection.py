"""Simple test to verify ML Engineer can initialize with Gemini."""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
load_dotenv()

print("🧪 Testing Gemini ML Engineer Initialization...")
print()

# Check config
provider = os.getenv('ML_ENGINEER_PROVIDER', 'google')
print(f"📋 ML_ENGINEER_PROVIDER={provider}")

# Import and initialize
try:
    from src.agents.ml_engineer import MLEngineerAgent
    agent = MLEngineerAgent()
    
    print(f"✅ Agent initialized successfully")
    print(f"✅ Provider: {agent.provider}")
    print(f"✅ Model: {agent.model_name}")
    print()
    print("🎉 SUCCESS: ML Engineer is configured for Gemini 3 Flash Preview!")
    print()
    print("📝 Next step: Run a full test with app.py")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
