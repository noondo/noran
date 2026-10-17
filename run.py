import sys
from production_app.main import run_training_workflow, run_prediction_workflow, run_workflow

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "train":
        run_training_workflow()
    elif mode == "predict":
        run_prediction_workflow()
    else:
        run_workflow()
