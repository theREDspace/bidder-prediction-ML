# Copilot Instructions for bid-prob-ML-system

## Project Overview
This project builds and exports a machine learning model to predict bidder participation for video ad inventory. The core logic is in `src/ML/`, with data processing, model training, and S3 export workflows.

## Key Components
- **dataframe.py**: Main script for data preparation, feature engineering, model training (LightGBM), and exporting artifacts.
- **s3_export.py**: Handles model and config export to AWS S3, including patching model files for Go compatibility.
- **config.json**: Stores feature list and bidder mapping for downstream consumers (e.g., Go services).
- **model.txt/model.dump**: Model artifacts for deployment.

## Data & Features
- Input: Hardcoded JSON objects representing ad requests and bidder responses.
- Features: width, height, aspect_ratio, is_mp4, min_dur, max_dur, bidder_id_encoded.
- Target: Whether a bidder participated in a given request.

## Model Training
- Uses LightGBM (`lgb.train`) for binary classification.
- Model is saved as `model.txt` (patched for Go if needed).
- `bidder_id` is encoded as categorical for modeling.

## Export Workflow
- After training, `export_to_s3` saves model and config locally, then uploads to S3 bucket `prebid-server-s3-bucket` under `bid-prediction/dev/`.
- Config includes feature list and bidder map for downstream use.
- Model file is patched from `version=v4` to `version=v3` for Go compatibility if detected.

## Developer Workflows
- **Run end-to-end**: Execute `dataframe.py` to train and export the model.
- **Dependencies**: Install from `requirements.txt` (pandas, scikit_learn, xgboost, lightgbm, boto3).
- **AWS Credentials**: Required for S3 export; errors are handled gracefully if missing.

## Project Conventions
- All features and bidder mappings must be kept in sync between model, config, and downstream consumers.
- Model and config filenames are hardcoded; update in both scripts if changed.
- Only LightGBM is used by default; XGBoost code is present but commented out.

## Integration Points
- S3 export is the main integration for model deployment.
- Downstream Go services expect patched model format and config structure.

## Example: Adding a Feature
1. Add feature extraction in `dataframe.py`.
2. Update `features` list in both code and config export.
3. Ensure downstream consumers are updated accordingly.

## References
- See `src/ML/dataframe.py` for end-to-end workflow.
- See `src/ML/s3_export.py` for export logic and S3 integration.
- See `src/ML/config.json` for current feature and bidder mapping.

