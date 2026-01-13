import boto3
import json
import lightgbm as lgb
import pandas as pd
import datetime
from botocore.exceptions import NoCredentialsError

# --- Configuration ---
BUCKET_NAME = 'prebid-server-s3-bucket' # TODO: Change this
S3_FOLDER = 'bid-prediction/dev'
MODEL_FILE = '/tmp/model.txt'
CONFIG_FILE = '/tmp/config.json'

def train_and_upload():
    # 1. Create Mock Data & Train (Same as before)
    # ---------------------------------------------------------
    print("⚡ Training LightGBM Model...")
    raw_data = [
        # --- Pattern: HD Content (1920x1080) ---
        # Bidder A usually wins these. Bidder C wins if it's MP4. Bidder B ignores them.
        {"id": "req_1", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"]},
        {"id": "req_2", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"]},
        {"id": "req_3", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"]}, # C hates webm
        {"id": "req_4", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"]},
        {"id": "req_5", "video": {"mimes": "video/mp4", "minduration": 30, "maxduration": 60, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"]},
        {"id": "req_6", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": []}, # No one bid (maybe A was out of budget)

        # --- Pattern: Mobile/Small Content (300x250) ---
        # Bidder B dominates here. Bidder A ignores.
        {"id": "req_7", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_8", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"]},
        {"id": "req_9", "video": {"mimes": "video/webm", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"]}, # C hates webm
        {"id": "req_10", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_11", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"]},
        {"id": "req_12", "video": {"mimes": "video/webm", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": []}, # B passed, C hates webm

        # --- Pattern: Vertical Video (720x1280 - Mobile App style) ---
        # Bidder B likes these too.
        {"id": "req_13", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_14", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB"]},
        {"id": "req_15", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB", "bidderC"]},

        # --- Pattern: Weird/Garbage Inventory ---
        # Odd sizes or constraints. Very low bid rate.
        {"id": "req_16", "video": {"mimes": "video/webm", "minduration": 60, "maxduration": 120, "w": 100, "h": 100}, "bidders": []},
        {"id": "req_17", "video": {"mimes": "video/mp4", "minduration": 0, "maxduration": 5, "w": 50, "h": 50}, "bidders": []},
        {"id": "req_18", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 300, "h": 250}, "bidders": ["bidderB"]}, # Standard mobile again
        {"id": "req_19", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"]}, # Standard HD again
        {"id": "req_20", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 300, "h": 250}, "bidders": ["bidderB"]},
    ]
    
    rows = []
    all_bidders = ["bidderA", "bidderB", "bidderC"]
    
    for req in raw_data:
        vid = req['video']
        winners = set(req['bidders'])
        base_feats = {
            'width': vid['w'], 'height': vid['h'],
            'aspect_ratio': vid['w']/vid['h'] if vid['h'] else 0,
            'is_mp4': 1 if 'mp4' in vid['mimes'] else 0,
            'min_dur': vid['minduration'], 'max_dur': vid['maxduration']
        }
        for bidder in all_bidders:
            row = base_feats.copy()
            row['bidder_id'] = bidder
            row['target'] = 1 if bidder in winners else 0
            rows.append(row)

    df = pd.DataFrame(rows)
    
    # Encode Bidders
    df['bidder_id_encoded'] = df['bidder_id'].astype('category').cat.codes
    bidder_map = {name: int(code) for name, code in zip(df['bidder_id'], df['bidder_id_encoded'])}
    
    # Features List (Order is critical for Go)
    features = ['bidder_id_encoded', 'width', 'height', 'aspect_ratio', 'is_mp4', 'min_dur', 'max_dur']
    
    # Train
    X = df[features]
    y = df['target']
    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=100,  # Small number of trees for small data
        min_child_samples=1,
        num_leaves=31,
        learning_rate=0.2,
        verbosity=-1
    )
    clf.fit(X, y)

    # 2. Export & Patch Model (CRITICAL STEP)
    # ---------------------------------------------------------
    clf.booster_.save_model(MODEL_FILE)
    
    # Patch v4 -> v3 for Go 'leaves' compatibility
    with open(MODEL_FILE, 'r') as f:
        content = f.read()
    
    if "version=v4" in content or "version=v5" in content:
        print("🔧 Patching model version to v3 for Go compatibility...")
        content = content.replace("version=v4", "version=v3").replace("version=v5", "version=v3")
        with open(MODEL_FILE, 'w') as f:
            f.write(content)

    # 3. Create Config JSON
    # ---------------------------------------------------------
    config_data = {
        "version": datetime.datetime.now().strftime("%Y%m%d-%H%M"),
        "features": features,
        "bidder_map": {"bidderA": 0, "bidderB": 1, "bidderC": 2} # e.g., {"bidderA": 0, "bidderB": 1}
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=2)

    # 4. Upload to S3
    # ---------------------------------------------------------
    s3 = boto3.client('s3')
    try:
        s3.upload_file(MODEL_FILE, BUCKET_NAME, S3_FOLDER + MODEL_FILE)
        s3.upload_file(CONFIG_FILE, BUCKET_NAME, S3_FOLDER + CONFIG_FILE)
        print(f"✅ Successfully uploaded to s3://{BUCKET_NAME}/{S3_FOLDER}")
    except NoCredentialsError:
        print("❌ Error: AWS Credentials not found")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    train_and_upload()