import cv2
import csv
import os
import mediapipe as mp
import numpy as np

print("STEP 1 SCRIPT STARTED")

mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose


def extract_features(frame, face_mesh, pose):
    """
    FIXED: Ab real values compute hoti hain — pehle sab zero tha!
    Yeh wahi features hain jis pe model train hua tha.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    features = {
        "face_detected":    0,
        "face_center_x":    0.5,
        "face_center_y":    0.5,
        "face_size":        0.0,
        "head_tilt":        0.0,
        "eye_contact_score":0.0,
        "shoulder_level":   0.0,
        "body_visible":     0,
        "mouth_open":       0.0,
    }

    # ── Face ──
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        features["face_detected"] = 1
        lm = face_results.multi_face_landmarks[0].landmark

        xs = [p.x for p in lm]
        ys = [p.y for p in lm]

        features["face_center_x"] = float(np.mean(xs))
        features["face_center_y"] = float(np.mean(ys))
        features["face_size"]     = float((max(xs)-min(xs)) * (max(ys)-min(ys)))

        # Head tilt — angle between left eye (33) and right eye (263)
        left_eye  = lm[33]
        right_eye = lm[263]
        dy = right_eye.y - left_eye.y
        dx = right_eye.x - left_eye.x
        features["head_tilt"] = float(abs(np.degrees(np.arctan2(dy, dx))))

        # Eye contact — how centered the face is
        cx = features["face_center_x"]
        cy = features["face_center_y"]
        features["eye_contact_score"] = float(
            max(0.0, 1.0 - (abs(cx - 0.5) * 2 + abs(cy - 0.4) * 2) / 2)
        )

        # Mouth open
        top_lip    = lm[13].y
        bottom_lip = lm[14].y
        features["mouth_open"] = float(abs(bottom_lip - top_lip))

    # ── Pose ──
    pose_results = pose.process(rgb)
    if pose_results.pose_landmarks:
        features["body_visible"] = 1
        pl = pose_results.pose_landmarks.landmark

        left_shoulder  = pl[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pl[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        features["shoulder_level"] = float(abs(left_shoulder.y - right_shoulder.y))

    return features


def collect_data(label, source=0, output_csv="dataset.csv", max_frames=300):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ CAMERA NOT OPENED")
        return
    else:
        print("✅ CAMERA OPENED")

    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)

    fieldnames = list(extract_features(
        np.zeros((480, 640, 3), dtype=np.uint8),
        mp_face_mesh.FaceMesh(),
        mp_pose.Pose()
    ).keys()) + ["label"]

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(output_csv).st_size == 0:
            writer.writeheader()

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh, mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            count = 0
            print(f"🎥 Recording {label.upper()}... Press Q to stop")

            while count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                features = extract_features(frame, face_mesh, pose)
                features["label"] = label
                writer.writerow(features)
                count += 1

                cv2.putText(frame, f"{label.upper()} {count}/{max_frames}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)

                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ DATA COLLECTION DONE")


if __name__ == "__main__":
    collect_data("good", max_frames=200)
    collect_data("bad",  max_frames=200)