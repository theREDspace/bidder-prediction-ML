import pandas as pd
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from s3_export import export_to_s3

# 1. Your raw input data (List of JSON objects)
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

# 2. Define your universe of bidders (Who are you inviting?)
# In production, this list comes from your partner configuration
eligible_bidders = ["bidderA", "bidderB", "bidderC"]

training_rows = []

for req in raw_data:
    video = req['video']
    participate_bidders = set(req['bidders']) # Use set for O(1) lookup
    
    print(participate_bidders)
    
    # Create features common to this request
    # Note: We handle mimes simply here, but you might need to parse lists if provided
    base_features = {
        "req_id": req['id'],
        "width": video['w'],
        "height": video['h'],
        "aspect_ratio": round(video['w'] / video['h'], 2) if video['h'] > 0 else 0,
        "is_mp4": 1 if "mp4" in video['mimes'] else 0,
        "min_dur": video['minduration'],
        "max_dur": video['maxduration']
    }

    # 3. The "Explosion": Create one row per eligible bidder
    for bidder in eligible_bidders:
        row = base_features.copy()
        row['bidder_id'] = bidder
        
        # TARGET VARIABLE: 1 if they are in the list, 0 otherwise
        row['target_bid'] = 1 if bidder in participate_bidders else 0
        
        training_rows.append(row)

# 4. Create the DataFrame
df = pd.DataFrame(training_rows)

# Encode bidder_id as categorical (Numbers) for the model
df['bidder_id_encoded'] = df['bidder_id'].astype('category').cat.codes

print(df[['req_id', 'bidder_id', 'width', 'is_mp4', 'target_bid']])

features = [
    'bidder_id_encoded', 
    'width', 
    'height', 
    'aspect_ratio', 
    'is_mp4', 
    'min_dur', 
    'max_dur'
]

X = df[features]
y = df['target_bid']

# ---------------------------------------------------------
# 2. SPLIT: Create Train and Test sets
# ---------------------------------------------------------
# Stratify ensures the ratio of Bids (1s) to No-Bids (0s) is consistent 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Create LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_data_in_leaf': 20
}

bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

# Export as Text (The format 'leaves' loves)
bst.save_model('model.txt')

# # ---------------------------------------------------------
# # 3. TRAIN: Initialize and Fit XGBoost
# # ---------------------------------------------------------
# # objective='binary:logistic' is CRITICAL. 
# # It tells XGBoost to minimize LogLoss and output probabilities.
# model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_estimators=100,      # Number of boosting rounds (trees)
#     learning_rate=0.1,     # Step size optimization
#     max_depth=5,           # Depth of trees (controls complexity)
#     use_label_encoder=False,
#     eval_metric='logloss'  # Metric to watch during training
# )

# model.fit(X_train, y_train)

# # ---------------------------------------------------------
# # 4. PREDICT: Generating Probabilities
# # ---------------------------------------------------------
# # .predict_proba() returns an array: [Prob_of_0, Prob_of_1]
# # We only want the second column (Prob_of_1)
# y_pred_probs = model.predict_proba(X_test)[:, 1]

# # ---------------------------------------------------------
# # 5. EVALUATE: How good are our probabilities?
# # ---------------------------------------------------------
# # Log Loss: The gold standard for probability accuracy. Lower is better.
# # If this is 0.69, your model is guessing. If < 0.10, it's very good.
# loss = log_loss(y_test, y_pred_probs)
# auc = roc_auc_score(y_test, y_pred_probs)

# print(f"Log Loss: {loss:.4f}")
# print(f"AUC Score: {auc:.4f}")

# # ---------------------------------------------------------
# # 6. INFERENCE: Example of a real prediction output
# # ---------------------------------------------------------
# # Let's look at the first 5 predictions vs actuals
# results = pd.DataFrame({
#     'Actual_Bid': y_test,
#     'Predicted_Probability': y_pred_probs
# })

# print("\n--- Prediction Samples ---")
# print(results.head())


bidder_map_export = {"bidderA": 0, "bidderB": 1, "bidderC": 2}

export_to_s3(bst, bidder_map_export, features)

