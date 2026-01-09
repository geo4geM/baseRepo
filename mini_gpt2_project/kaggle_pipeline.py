"""Kaggle entry point for running Mini-GPT2 experiments in Kaggle environments."""

"""
Mini-GPT2 Track B: Kaggle Pipeline
"""
import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "-q"])

def run_kaggle_pipeline():
    print("Starting Kaggle Pipeline...")
    install_dependencies()
    
    # Setup paths
    sys.path.insert(0, str(Path(__file__).parent))
    from main import main
    
    # Trigger main execution
    main()

if __name__ == "__main__":
    run_kaggle_pipeline()