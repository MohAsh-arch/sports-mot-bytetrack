import cv2
import json
import numpy as np
from pathlib import Path
from src.homography import DEFAULT_PITCH_POINTS

# Points the user needs to click (in order)
TARGET_POINTS = [
    ("top_left_penalty_box", DEFAULT_PITCH_POINTS["top_left_penalty_box"]),
    ("top_right_penalty_box", DEFAULT_PITCH_POINTS["top_right_penalty_box"]),
    ("bottom_right_penalty_box", DEFAULT_PITCH_POINTS["bottom_right_penalty_box"]),
    ("bottom_left_penalty_box", DEFAULT_PITCH_POINTS["bottom_left_penalty_box"])
]

clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"✅ Clicked: ({x}, {y})")
            
            # Draw a dot where the user clicked
            img_copy = param.copy()
            for pt in clicked_points:
                cv2.circle(img_copy, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.imshow("Calibration", img_copy)

def run_calibration(video_path):
    print("⚽ Starting Pitch Calibration ⚽")
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read video.")
        return

    print("\n👉 Please click the following 4 points on the image IN THIS EXACT ORDER:")
    for i, (name, _) in enumerate(TARGET_POINTS):
        print(f"  {i+1}. {name.replace('_', ' ').title()}")
    
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", click_event, frame)
    
    print("\nPress any key after clicking 4 points, or press 'q' to quit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(clicked_points) == 4:
        # Save to JSON
        dst_pts = [pt for _, pt in TARGET_POINTS]
        data = {
            "src_pts": clicked_points,
            "dst_pts": dst_pts
        }
        with open("pitch_keypoints.json", "w") as f:
            json.dump(data, f, indent=2)
        print("\n🎉 Success! pitch_keypoints.json saved.")
        print("You can now enable the Offside Line and Defensive Triangle in the Streamlit app!")
    else:
        print(f"\n❌ Calibration failed. You clicked {len(clicked_points)} points instead of 4.")

if __name__ == "__main__":
    # You can change this to your specific video path
    run_calibration("img1.mp4")
