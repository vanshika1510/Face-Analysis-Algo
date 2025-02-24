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

if analysis_results.get("Blackheads", 0) > 50:
    print("https://www.amazon.in/dp/B0DCC1YHCL/ref=sspa_dk_detail_2?pd_rd_i=B0DCC1YHCL&pd_rd_w=yfGQl&content-id=amzn1.sym.9f1cb690-f0b7-44de-b6ff-1bad1e37d3f0&pf_rd_p=9f1cb690-f0b7-44de-b6ff-1bad1e37d3f0&pf_rd_r=JFZJ2PQ33KQDG3T4J2NB&pd_rd_wg=dpkyJ&pd_rd_r=4ebb76ce-8c79-4d9d-bd9d-d8b78b06f5f0&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&th=1")
if analysis_results.get("Wrinkles", 0) > 50:
    print("https://amzn.in/d/csCZKbS")
if analysis_results.get("Pores", 0) > 50:
    print("https://www.amazon.in/Lacto-Calamine-Niacinamide-minimising-Dermatologically/dp/B0BNQW9JV9/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.fDcfFGPSNnKbQ5Fm6OTgpqI2HBk9d9R8gM__tuZnS4-uiLsSxZxiqyWccBnnBLEeJi0GirBmntdyuJjUVI2kyYs2KGJNR_dgy2aGdCNMo7hARHwiZXTz9DM2MBeLVdE2bbmUefJzAf9sDenPzmzJZmbWrhStFs1N3IJzzAat3nVVxkt4DUcLcO88PYQ0d85Yz3OL_J4tQJEs4rT15ZudP9Nz-sKnzrA7JHETOHLsDQjpa5anI4ZTg_rNi0EK5Th2I3Wlm_jt4a2vLZVvRHCYtlg3L0v-ah_Dgf-QMY7EW_Y.5fxcfQSM6a-si9yNkfPg-rWG2KssMkcnysTtB1Gu9kk&dib_tag=se&keywords=best+cream+for+open+pores+on+face&qid=1740388894&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1")
if analysis_results.get("Eye Bags", 0) > 50:
    print("https://amzn.in/d/aqe5ZTQ")

print("\nSkin Analysis Results:")
for condition, confidence in analysis_results.items():
    print(f"{condition}: {confidence:.2f}%")

print("\nBest Matching URL:", best_url)
