"""Quick smoke test — verifies all imports work."""
import sys
sys.path.insert(0, ".")

errors = []

modules = [
    ("config", "from config import DEVICE, OUTPUT_DIR"),
    ("src.detection", "from src.detection import PlayerDetector, detect_video_mode"),
    ("src.tracking", "from src.tracking import ByteTrackWrapper, SORT, run_tracking_loop, run_sort_baseline"),
    ("src.ball", "from src.ball import BallDetector, BallKalmanGate, run_ball_detection"),
    ("src.teams", "from src.teams import classify_teams"),
    ("src.possession", "from src.possession import compute_possession"),
    ("src.heatmap", "from src.heatmap import generate_team_heatmaps, generate_ball_trajectory"),
    ("src.video", "from src.video import render_annotated_video"),
    ("src.evaluation", "from src.evaluation import compare_trackers"),
    ("src.homography", "from src.homography import calibrate_homography, get_homography"),
    ("src.overlays", "from src.overlays import draw_all_overlays"),
]

for name, stmt in modules:
    try:
        exec(stmt)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        errors.append(name)

print()
if errors:
    print(f"FAILED: {len(errors)} modules had import errors: {errors}")
    sys.exit(1)
else:
    print("ALL IMPORTS OK ✓")
    from config import DEVICE
    print(f"  Device: {DEVICE}")
