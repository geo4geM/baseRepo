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

    # Setup paths so `mini_gpt2_project` is imported as a proper package.
    # We add the *parent* of this directory to sys.path and then import
    # `mini_gpt2_project.main`. This ensures that relative imports such as
    # `from ..config.model_config import ...` inside submodules work correctly.
    project_root = Path(__file__).parent  # .../mini_gpt2_project
    repo_root = project_root.parent       # repository root
    sys.path.insert(0, str(repo_root))

    from mini_gpt2_project.main import main
    
    # Trigger main execution
    main()

if __name__ == "__main__":
    run_kaggle_pipeline()