import argparse
import os
import sys
from pathlib import Path

# Redirect stdout to stderr immediately to prevent logger output in stdout
original_stdout = sys.stdout
sys.stdout = sys.stderr

# Configure legacy Keras before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Add project root to sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.pipeline.predict import NLPPredictor

# Patch the load_model method for BERT
def patched_load_model(self):
    from src.models.bert_model import BERTModel
    bert_builder = BERTModel()
    model = bert_builder.build_model()
    model.load_weights(str(self.model_path))
    return model

NLPPredictor._load_model = patched_load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    
    predictor = NLPPredictor(model_type="bert")
    result = predictor.predict(args.text)
    
    import json
    # Print only the JSON result to the original stdout
    print(json.dumps(result), file=original_stdout)
