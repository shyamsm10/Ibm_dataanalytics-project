"""
Training entry-point script.
Run from the project root: python src/train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_trainer import train_all_models

if __name__ == "__main__":
    print("=" * 50)
    print("Cafe Sales Prediction - Model Training")
    print("=" * 50)
    results = train_all_models()
    print("\nTraining complete.")
    print(f"Best model: {results['best_model']}")
    best = results[results['best_model']]
    print(f"  MAE  : {best['MAE']}")
    print(f"  RMSE : {best['RMSE']}")
    print(f"  R2   : {best['R2']}")
    print("Model saved to models/best_model.pkl")
    print("Metrics saved to models/model_metrics.json")
