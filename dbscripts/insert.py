import boto3
import time

# 1. Initialize the DynamoDB Resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('PrebidBidReports')

# 2. Your JSON object (as a Python dict)
data = [
        # --- Pattern: HD Content (1920x1080) ---
        # Bidder A usually wins these. Bidder C wins if it's MP4. Bidder B ignores them.
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"], "auctiontimestamp": int(time.time() * 1000)}, # C hates webm
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 30, "maxduration": 60, "w": 1920, "h": 1080}, "bidders": ["bidderA", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": [], "auctiontimestamp": int(time.time() * 1000)}, # No one bid (maybe A was out of budget)

        # --- Pattern: Mobile/Small Content (300x250) ---
        # Bidder B dominates here. Bidder A ignores.
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/webm", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"], "auctiontimestamp": int(time.time() * 1000)}, # C hates webm
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": ["bidderB"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/webm", "minduration": 5, "maxduration": 15, "w": 300, "h": 250}, "bidders": [], "auctiontimestamp": int(time.time() * 1000)}, # B passed, C hates webm

        # --- Pattern: Vertical Video (720x1280 - Mobile App style) ---
        # Bidder B likes these too.
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB"], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 10, "maxduration": 20, "w": 720, "h": 1280}, "bidders": ["bidderB", "bidderC"], "auctiontimestamp": int(time.time() * 1000)},

        # --- Pattern: Weird/Garbage Inventory ---
        # Odd sizes or constraints. Very low bid rate.
        {"id": "SiteID_123", "video": {"mimes": "video/webm", "minduration": 60, "maxduration": 120, "w": 100, "h": 100}, "bidders": [], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 0, "maxduration": 5, "w": 50, "h": 50}, "bidders": [], "auctiontimestamp": int(time.time() * 1000)},
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 300, "h": 250}, "bidders": ["bidderB"], "auctiontimestamp": int(time.time() * 1000)}, # Standard mobile again
        {"id": "SiteID_123", "video": {"mimes": "video/mp4", "minduration": 15, "maxduration": 30, "w": 1920, "h": 1080}, "bidders": ["bidderA"], "auctiontimestamp": int(time.time() * 1000)}, # Standard HD again
        {"id": "SiteID_123", "video": {"mimes": "video/webm", "minduration": 15, "maxduration": 30, "w": 300, "h": 250}, "bidders": ["bidderB"], "auctiontimestamp": int(time.time() * 1000)},
    ]

# 3. Save to DynamoDB
try:
    for entry in data:
        entry["auctiontimestamp"] = int(time.time() * 1000)
        response = table.put_item(Item=entry)
        time.sleep(0.5)
    print("Item saved successfully!")
except Exception as e:
    print(f"Error saving to DynamoDB: {e}")