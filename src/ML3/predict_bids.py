import pandas as pd
import lightgbm as lgb
import json

class BidPredictionSystem:
    def __init__(self):
        self.model = None
        self.bidder_map = {}
        # Exact feature order for consistency
        self.feature_columns = [
            'bidder_id_encoded', 'width', 'height', 
            'aspect_ratio', 'is_mp4', 'min_dur', 'max_dur'
        ]

    def train_with_raw_data(self, raw_data):
        print("⚡ Training model on provided raw_data...")
        
        all_bidders = ["bidderA", "bidderB", "bidderC"]
        rows = []

        # 1. DATA TRANSFORMATION (Explode Request -> Rows)
        for req in raw_data:
            vid = req['video']
            winners = set(req['bidders'])
            
            # Feature Engineering
            base_feats = {
                'width': vid['w'], 
                'height': vid['h'],
                'aspect_ratio': vid['w'] / vid['h'] if vid['h'] > 0 else 0,
                'is_mp4': 1 if 'mp4' in vid['mimes'] else 0,
                'min_dur': vid['minduration'], 
                'max_dur': vid['maxduration']
            }
            
            # Create a row for EVERY eligible bidder (Positive & Negative samples)
            for bidder in all_bidders:
                row = base_feats.copy()
                row['bidder_id'] = bidder
                row['target'] = 1 if bidder in winners else 0
                rows.append(row)

        df = pd.DataFrame(rows)
        
        # 2. ENCODING
        df['bidder_id_encoded'] = df['bidder_id'].astype('category').cat.codes
        
        # Save map: {'bidderA': 0, 'bidderB': 1, ...}
        categories = df['bidder_id'].astype('category').cat.categories
        self.bidder_map = {name: i for i, name in enumerate(categories)}
        
        # 3. TRAIN
        X = df[self.feature_columns]
        y = df['target']
        
        # Using LightGBM for speed and accuracy
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=100,  # Small number of trees for small data
            min_child_samples=1,
            num_leaves=31,
            learning_rate=0.2,
            verbosity=-1
        )
        self.model.fit(X, y)
        print("✅ Model trained successfully.")

    def predict(self, new_request, candidates):
        if not self.model: return {}

        vid = new_request['video']
        base_features = {
            'width': vid['w'],
            'height': vid['h'],
            'aspect_ratio': vid['w'] / vid['h'] if vid['h'] > 0 else 0,
            'is_mp4': 1 if 'mp4' in vid['mimes'] else 0,
            'min_dur': vid['minduration'],
            'max_dur': vid['maxduration']
        }
        
        inference_rows = []
        valid_bidders = []
        
        for bidder in candidates:
            if bidder in self.bidder_map:
                row = base_features.copy()
                row['bidder_id_encoded'] = self.bidder_map[bidder]
                inference_rows.append(row)
                valid_bidders.append(bidder)
        
        if not inference_rows: return {}

        df_inf = pd.DataFrame(inference_rows)
        # Predict Probability (Column 1)
        probs = self.model.predict_proba(df_inf[self.feature_columns])[:, 1]
        
        return dict(zip(valid_bidders, probs))

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Your Provided Data
    raw_data = [
        {"id": "req_1", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"]},
        {"id": "req_2", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"]},
        {"id": "req_3", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"]},
        {"id": "req_4", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"]},
        {"id": "req_5", "video": {"mimes": "video/mp4", "minduration": 30, "maxduration": 60, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"]},
        {"id": "req_6", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": []},
        {"id": "req_7", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_8", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"]},
        {"id": "req_9", "video": {"mimes": "video/webm", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"]},
        {"id": "req_10", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_11", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"]},
        {"id": "req_12", "video": {"mimes": "video/webm", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": []},
        {"id": "req_13", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_14", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB"]},
        {"id": "req_15", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB", "bidderC"]},
        {"id": "req_16", "video": {"mimes": "video/webm", "minduration": 60, "maxduration": 120, "w": 100, "h": 100}, "bidders": []},
        {"id": "req_17", "video": {"mimes": "video/mp4", "minduration": 0, "maxduration": 5, "w": 50, "h": 50}, "bidders": []},
        {"id": "req_18", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 300, "h": 250}, "bidders": ["bidderB"]},
        {"id": "req_19", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"]},
        {"id": "req_20", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 300, "h": 250}, "bidders": ["bidderB"]},
    ]

    # 2. Train System
    system = BidPredictionSystem()
    system.train_with_raw_data(raw_data)

    # 3. PREDICTION TESTS
    # We create scenarios to verify the model learned the "personalities"
    
    test_cases = [
        {
            "desc": "HD MP4 (BidderA's favorite, BidderC likes MP4)",
            "req": {"id": "test_1", "video": {"mimes": "video/webm", "w": 1920, "h": 1080, "minduration": 15, "maxduration": 30}}
        },
        {
            "desc": "Mobile WebM (BidderB likes size, BidderC hates WebM)",
            "req": {"id": "test_2", "video": {"mimes": "video/mp4", "w": 300, "h": 250, "minduration": 5, "maxduration": 15}}
        },
        {
            "desc": "Weird Tiny Video (Everyone should ignore)",
            "req": {"id": "test_3", "video": {"mimes": "video/mp4", "w": 50, "h": 50, "minduration": 0, "maxduration": 5}}
        }
    ]

    candidates = ["bidderA", "bidderB", "bidderC"]

    print("\n" + "="*60)
    print(f"{'SCENARIO':<40} | {'BIDDER':<8} | {'PROB':<8}")
    print("="*60)

    for case in test_cases:
        probs = system.predict(case['req'], candidates)
        print(f"\n🔹 {case['desc']}")
        for bidder, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            # Highlight likely bidders
            mark = "✅" if p > 0.5 else "❌"
            print(f"{'':<40} | {bidder:<8} | {p:.4f} {mark}")