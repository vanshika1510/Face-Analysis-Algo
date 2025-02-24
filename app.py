import sys
import os
import cv2
import numpy as np
import pandas as pd
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="hiNFrsKV6b1wmbyc1i04"
)

image_path = r"test.png"  # Replace with your actual image path
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    sys.exit(1)

def split_and_infer(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    step_x = width // 5  # Divide into 3 vertical parts
    step_y = height // 5  # Divide into 3 horizontal parts
    
    max_confidences = {}
    
    for i in range(5):
        for j in range(5):
            x1, y1 = i * step_x, j * step_y
            x2, y2 = (i + 1) * step_x, (j + 1) * step_y
            cropped_image = image[y1:y2, x1:x2]
            temp_path = "temp_crop.jpg"
            cv2.imwrite(temp_path, cropped_image)
            
            try:
                result = CLIENT.infer(temp_path, model_id="skin-problem-multilabel/1")
                predictions = result.get("predictions", {})
                
                for condition, data in predictions.items():
                    confidence = data["confidence"] * 100
                    max_confidences[condition] = max(max_confidences.get(condition, 0), confidence)
                
            except Exception as e:
                print(f"Error during inference for segment ({i}, {j}): {e}")
    
    return max_confidences

analysis_results = split_and_infer(image_path)

def get_range(confidence):
    if confidence <= 25:
        return 25
    elif confidence <= 50:
        return 50
    elif confidence <= 75:
        return 75
    else:
        return 100

selected_conditions = ["Acne", "Dark Spots", "Oily Skin", "Dry Skin", "Skin Redness"]
test_case = {condition: get_range(analysis_results.get(condition, 0)) for condition in selected_conditions}

file_path = "products.csv"  # Ensure correct path
df = pd.read_csv(file_path)

def find_best_match(df, test_case):
    df["match_score"] = df.apply(lambda row: sum(row[col] == test_case[col] for col in test_case.keys()), axis=1)
    best_match = df.loc[df["match_score"].idxmax()]
    return best_match["URL"]

best_url = find_best_match(df, test_case)

print("\nSkin Analysis Results:")
for condition, confidence in analysis_results.items():
    print(f"{condition}: {confidence:.2f}%")

print("\nBest Matching URL:", best_url)
