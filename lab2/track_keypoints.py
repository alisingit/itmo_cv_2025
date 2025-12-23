import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def build_detector():
    return cv2.SIFT_create(nfeatures=5000, nOctaveLayers=5,         # Увеличиваем слоев в октаве (по умолчанию 3)
    contrastThreshold=0.01,  # Уменьшаем порог контраста (по умолчанию 0.04)
    edgeThreshold=15,        # Увеличиваем порог границ (по умолчанию 10)
    sigma=0.8               )
    # return cv2.ORB_create(
    # nfeatures=3000,           # Больше точек для детального изображения
    # scaleFactor=1.05,         # Мелкий шаг для плавного отслеживания масштаба
    # nlevels=16,               # Много уровней для большого диапазона масштабов
    # edgeThreshold=10,         # Картина может быть до краев, учитываем края
    # firstLevel=0,             # Начинаем с полного размера
    # WTA_K=2,                  # Стандарт для ORB
    # scoreType=cv2.ORB_HARRIS_SCORE,  # Лучшая пространственная локализация
    # patchSize=31,             # Стандарт для детальных текстур
    # fastThreshold=7           # Низкий порог для учета всех деталей картины
# )


def build_matcher():
    return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def extract_features(detector, gray: np.ndarray, mask: np.ndarray = None):
    keypoints, desc = detector.detectAndCompute(gray, mask)
    return keypoints, desc


def match_features(matcher, desc_obj, desc_frame, ratio: float = 0.75):
    if desc_obj is None or desc_frame is None:
        return []

    matches = matcher.knnMatch(desc_obj, desc_frame, k=2)
    good = []
    try:
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
    except:
        return good
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
    last_corners = None
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
    if not last_corners is None:
        corners = np.float32(last_corners).reshape(-1, 1, 2)

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
    last_corners = None
    last_mask = None
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_frame, desc_frame = extract_features(detector, gray, None)
        good_matches = match_features(matcher, desc_obj, desc_frame)

        H, mask = compute_homography(kp_obj, kp_frame, good_matches, min_matches)

        if H is not None:
            last_mask_temp, corners_temp = get_mask(H, tmpl_h, tmpl_w, last_corners)
            draw_object_bbox(frame, H, (tmpl_h, tmpl_w), label="object", last_corners=last_corners)
            lost_frames = 0
            
            
            sz_1 = np.count_nonzero(last_mask_temp == 255)
            sz_2 = np.count_nonzero(last_mask == 255)
            
            if not last_mask is None:
                print(f'Size Diff {(min(sz_1,sz_2) / max(sz_1,sz_2))}')  
                print(f'Form Diff {mask_difference_percent(last_mask, last_mask_temp)}')
            
            if last_mask is None: #or (min(sz_1,sz_2) / max(sz_1,sz_2)) > 0.8 and mask_difference_percent(last_mask, last_mask_temp) < 0.1:
                #print(f'Curr: {sz_1} Last: {sz_2}')
                last_mask = last_mask_temp
                #kp_obj, desc_obj = extract_features(detector, tmpl_gray, last_mask)
                continue
            if check_opposite_angles_equal(corners_temp) and (min(sz_1,sz_2) / max(sz_1,sz_2)) <= 0.4 and (min(sz_1,sz_2) / max(sz_1,sz_2)) >= 0.2:
            #if check_opposite_angles_equal(corners_temp, 15) and ((1 - mask_difference_percent(last_mask, last_mask_temp)) >= 0.6 and (1 - mask_difference_percent(last_mask, last_mask_temp)) <= 0.8 or (min(sz_1,sz_2) / max(sz_1,sz_2)) <= 0.4 and (min(sz_1,sz_2) / max(sz_1,sz_2)) >= 0.2):
                last_mask = last_mask_temp
                kp_obj, desc_obj = extract_features(detector, gray, last_mask)
                #cv2.imwrite("tetst.jpg", cv2.bitwise_and(gray, last_mask))
                # print(f'Size Diff {(min(sz_1,sz_2) / max(sz_1,sz_2))}')  
                # print(f'Form Diff {1 - mask_difference_percent(last_mask, last_mask_temp)}')
                last_corners = corners_temp
                
                #tmpl_h, tmpl_w = abs(corners_temp[0] - corners_temp[1]) , abs(corners_temp[1] - corners_temp[2])
        else:
            #cv2.polylines(frame, [last_mask_cornerns], isClosed=True, color=(255, 255, 0), thickness=5)
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

def get_bbox_from_corners(corners):
    # Преобразуем в удобный формат
    pts = corners.reshape(4, 2)
    
    # Находим min/max координаты
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Bounding box: (x, y, width, height)
    x = int(x_min)
    y = int(y_min)
    w = int(x_max - x_min)
    h = int(y_max - y_min)
    
    return w, h
def check_opposite_angles_equal(corners, angle_tolerance=15.0):
    """
    Проверяет, что противоположные углы четырехугольника примерно равны
    
    Args:
        corners: массив углов [[[x1,y1]], [[x2,y2]], [[x3,y3]], [[x4,y4]]]
        angle_tolerance: допустимое отклонение углов в градусах
    
    Returns:
        dict: результаты проверки
    """
    # Преобразуем к формату (4, 2)
    pts = corners.reshape(4, 2)
    
    # Вычисляем все углы четырехугольника
    angles = []
    for i in range(4):
        # Точка угла
        p = pts[i]
        # Предыдущая и следующая точки
        prev_p = pts[(i-1) % 4]
        next_p = pts[(i+1) % 4]
        
        # Векторы от угла к соседним точкам
        v1 = prev_p - p
        v2 = next_p - p
        
        # Вычисляем угол между векторами
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)  # Защита от численных ошибок
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)
    
    # Противоположные углы:
    # Угол 0 ↔ Угол 2
    # Угол 1 ↔ Угол 3
    
    diff_0_2 = abs(angles[0] - angles[2])
    diff_1_3 = abs(angles[1] - angles[3])
    
    # Проверяем равенство
    is_0_2_equal = diff_0_2 < angle_tolerance
    is_1_3_equal = diff_1_3 < angle_tolerance
    
    # Общая оценка
    is_valid = is_0_2_equal and is_1_3_equal
    
    return is_valid
      
def mask_difference_percent(mask1, mask2):
    """
    Коэффициент Жаккара (IoU)
    1.0 = идентичные, 0.0 = нет пересечения
    """
    binary1 = (mask1 > 0).astype(bool)
    binary2 = (mask2 > 0).astype(bool)
    
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union
def get_mask(H,tmpl_h,tmpl_w , cornersP):
    corners = np.float32(
        [
            [0, 0],
            [tmpl_w - 1, 0],
            [tmpl_w - 1, tmpl_h - 1],
            [0, tmpl_h - 1],
        ]
    ).reshape(-1, 1, 2)
    if not cornersP is None:
        corners = np.float32(cornersP).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, H)

    dst_int = np.int32(dst)
    
    object_mask = np.zeros((tmpl_h, tmpl_w), dtype=np.uint8)
    # 3. Заполняем полигон объекта
    cv2.fillPoly(object_mask, [dst_int], color=255)
    # print(tmpl_h, tmpl_w)
    cv2.imwrite('smth.jpg', object_mask)
    return object_mask, dst_int
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
