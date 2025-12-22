import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def build_detector():
    return cv2.ORB_create(nfeatures=1000)


def build_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def extract_features(detector, gray: np.ndarray):
    keypoints, desc = detector.detectAndCompute(gray, None)
    return keypoints, desc


def match_features(matcher, desc_obj, desc_frame, ratio: float = 0.75):
    if desc_obj is None or desc_frame is None:
        return []

    matches = matcher.knnMatch(desc_obj, desc_frame, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def compute_homography(
    kp_obj, kp_frame, matches, min_matches: int = 10
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(matches) < min_matches:
        return None, None

    src_pts = np.float32([kp_obj[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask


def draw_object_bbox(
    frame: np.ndarray,
    H: np.ndarray,
    template_size: Tuple[int, int],
    label: str = "object",
):
    h, w = template_size

    corners = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    ).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(corners, H)

    dst_int = np.int32(dst)
    cv2.polylines(frame, [dst_int], isClosed=True, color=(0, 255, 0), thickness=2)

    x, y = dst_int[0, 0]
    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def process_video(
    path: str,
    output_path: Optional[str] = None,
    min_matches: int = 12,
    show: bool = True,
):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {path}")
        return

    ok, first_frame = cap.read()
    if not ok:
        print("Первый кадр прочитать не получилось.")
        cap.release()
        return

    tmpl_h, tmpl_w = first_frame.shape[:2]
    tmpl_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    detector = build_detector()
    matcher = build_matcher()

    kp_obj, desc_obj = extract_features(detector, tmpl_gray)

    if desc_obj is None or len(kp_obj) == 0:
        print("На первом кадре не удалось найти ключевые точки.")
        cap.release()
        return

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25.0  # fallback
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    frame_idx = 0
    lost_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_frame, desc_frame = extract_features(detector, gray)
        good_matches = match_features(matcher, desc_obj, desc_frame)

        H, mask = compute_homography(kp_obj, kp_frame, good_matches, min_matches)

        if H is not None:
            draw_object_bbox(frame, H, (tmpl_h, tmpl_w), label="object")
            lost_frames = 0
        else:
            lost_frames += 1
            cv2.putText(
                frame,
                "tracking lost",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if writer is not None:
            writer.write(frame)

        if show:
            cv2.imshow("tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ЛР2: трекинг объекта на основе ключевых точек (ORB + homography)."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Путь к входному видео. На первом кадре объект крупным планом.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Путь к выходному видео (mp4). Если не задан, видео не сохраняется.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=12,
        help="Минимальное число хороших совпадений для построения гомографии.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не показывать окно с видео, только обработка/сохранение.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Файл не найден: {args.input}")
        sys.exit(1)

    process_video(
        path=args.input,
        output_path=args.output,
        min_matches=args.min_matches,
        show=not args.no_show,
    )
