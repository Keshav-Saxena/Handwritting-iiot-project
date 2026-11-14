import json
import os

def publish_result(digit, confidence):
    # Data ko ek dictionary mein rakhein
    message = {
        "digit": digit,
        "confidence": confidence
    }
    
    try:
        # Results folder ya current directory mein ek JSON file save karein
        os.makedirs("results", exist_ok=True)
        file_path = "results/latest_digit.json"
        
        with open(file_path, "w") as f:
            json.dump(message, f)
            
        return message, True, "Saved locally successfully"
    except Exception as e:
        return message, False, f"Failed to save: {e}"