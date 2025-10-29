# realtime_test.py
import cv2
from core.skin_analysis import analyze_image

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return

    print("✅ Webcam started. Press 'q' to quit and generate final PDF report.")
    last_frame_path = "last_frame.jpg"
    last_results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Display instructions on screen
        cv2.putText(frame, "Press 'q' to quit & generate PDF report",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        cv2.imshow("📸 Real-Time Skin Analysis - DD AURA", frame)

        # Wait for key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # Save last frame before quitting
            cv2.imwrite(last_frame_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Run analysis once on the last frame AFTER quit
    print("🔍 Analyzing last captured frame...")
    last_results = analyze_image(last_frame_path)

    if last_results:
        print("✅ Final PDF report generated.")
        for k, v in last_results.items():
            print(f"{k}: {v}")
    else:
        print("⚠️ No results generated.")

if __name__ == "__main__":
    main()
