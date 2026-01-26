import boto3
import json
import lightgbm as lgb
import pandas as pd
import datetime
from botocore.exceptions import NoCredentialsError
from boto3.dynamodb.conditions import Key

# --- Configuration ---
BUCKET_NAME = 'prebid-server-s3-bucket' # TODO: Change this
S3_FOLDER = 'bid-prediction/dev'
MODEL_FILE = '/tmp/model.txt'
CONFIG_FILE = '/tmp/config.json'

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('PrebidBidReports')

def get_latest_auctions(partition_key_value):
    response = table.query(
        # You must provide the Partition Key to use Query
        KeyConditionExpression=Key('id').eq(partition_key_value),
        
        # This sorts by auctiontimestamp in descending order
        ScanIndexForward=False, 
        
        # Limit the result set to 50
        Limit=50
    )
    
    return response.get('Items', [])

def train_and_upload():
    # 1. Create Mock Data & Train (Same as before)
    # ---------------------------------------------------------
    print("⚡ Training LightGBM Model...")
    raw_data = get_latest_auctions('SiteID_123')  # Example Partition Key Value
    
    rows = []
    all_bidders = ["bidderA", "bidderB", "bidderC"]
    
    for req in raw_data:
        vid = req['video']
        winners = set(req['bidders'])
        base_feats = {
            'width': float(vid['w']), 'height': float(vid['h']),
            'aspect_ratio': float(vid['w'])/float(vid['h']) if vid['h'] else 0,
            'is_mp4': 1 if 'mp4' in vid['mimes'] else 0,
            'min_dur': float(vid['minduration']), 'max_dur': float(vid['maxduration'])
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