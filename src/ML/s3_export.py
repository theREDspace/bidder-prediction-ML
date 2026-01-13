import boto3
import json
import os
import datetime
from botocore.exceptions import NoCredentialsError

# 1. Setup: Define your S3 details
BUCKET_NAME = 'prebid-server-s3-bucket'
S3_FOLDER = 'bid-prediction/dev/'
MODEL_FILENAME = 'model.txt'
CONFIG_FILENAME = 'config.json'

def export_to_s3(bst, bidder_map, feature_list):
    # --- A. Save Model Locally ---
    bst.save_model("model.txt")
    # model.get_booster().save_model(MODEL_FILENAME)
    
    with open('model.txt', 'r') as f:
        content = f.read()
        
    if "version=v4" in content:
        print("⚠️  Patching model version from v4 to v3 for Go compatibility...")
        content = content.replace("version=v4", "version=v3")
        # Save it back
        with open('model.txt', 'w') as f:
            f.write(content)
            
    print("✅ Exported model.txt (Patched to v3)")
    
    # --- B. Save Config/Metadata Locally ---
    # This ensures Go knows exactly how to encode features
    config_data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "features": feature_list,
        "bidder_map": bidder_map,  # e.g., {'bidderA': 0, 'bidderB': 1}
        "version": "v1.2.0"
    }
    
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(config_data, f, indent=2)

    # --- C. Upload to AWS S3 ---
    s3 = boto3.client('s3')

    try:
        # Upload Model
        s3.upload_file(
            MODEL_FILENAME, 
            BUCKET_NAME, 
            f"{S3_FOLDER}{MODEL_FILENAME}"
        )
        print(f"✅ Model uploaded to s3://{BUCKET_NAME}/{S3_FOLDER}{MODEL_FILENAME}")

        # Upload Config
        s3.upload_file(
            CONFIG_FILENAME, 
            BUCKET_NAME, 
            f"{S3_FOLDER}{CONFIG_FILENAME}"
        )
        print(f"✅ Config uploaded to s3://{BUCKET_NAME}/{S3_FOLDER}{CONFIG_FILENAME}")

    except NoCredentialsError:
        print("❌ Error: AWS credentials not found.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
