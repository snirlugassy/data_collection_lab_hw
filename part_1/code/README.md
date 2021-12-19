# Data collection and management - Homework 1

installing dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

### Step 1 - Training

```
python train.py <training-data-file-path>
```

### Step 1 - Inference (Exporting JSON with words)

```
python generate_snippets.py <unlabeled-data-file-path> <trained-model-file-path>
```

The trained model file path will be `model.sklearn` in order case

The output file will be `output.json`