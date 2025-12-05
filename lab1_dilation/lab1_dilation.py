import argparse
import os
import time

import cv2
import numpy as np


def binarize(gray, threshold=128):
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def dilate_cv(binary):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def dilate_naive(binary):
    h, w = binary.shape
    result = np.zeros_like(binary, dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            window = binary[y - 1:y + 2, x - 1:x + 2]
            if np.any(window):
                result[y, x] = 255

    return result


def measure_time(fn, img, repeats=5):
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        _ = fn(img)
        total += time.perf_counter() - start
    return (total / repeats) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='ЛР1. Дилатация бинарного изображения (вариант 6).'
    )
    parser.add_argument(
        '-i', '--image',
        required=True,
        help='Путь к входному изображению (цветное или grayscale).'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=128,
        help='Порог бинаризации (по умолчанию 128).'
    )
    parser.add_argument(
        '-r', '--repeats',
        type=int,
        default=5,
        help='Количество повторов для замера времени (по умолчанию 5).'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f'Не удалось открыть изображение: {args.image}')
        exit(1)

    bin_img = binarize(gray, threshold=args.threshold)

    dil_cv = dilate_cv(bin_img)
    dil_naive = dilate_naive(bin_img)

    t_cv = measure_time(dilate_cv, bin_img, repeats=args.repeats)
    t_naive = measure_time(dilate_naive, bin_img, repeats=args.repeats)

    print('=== Замер времени (среднее) ===')
    print(f'OpenCV:   {t_cv:.3f} мс (при {args.repeats} повторах)')
    print(f'Naive:    {t_naive:.3f} мс (при {args.repeats} повторах)')

    base = os.path.splitext(os.path.basename(args.image))[0]

    out_bin = f'{base}_binary.png'
    out_cv = f'{base}_dilate_cv.png'
    out_naive = f'{base}_dilate_naive.png'

    cv2.imwrite(out_bin, bin_img)
    cv2.imwrite(out_cv, dil_cv)
    cv2.imwrite(out_naive, dil_naive)

    print('\nФайлы сохранены:')
    print(f'  бинаризация: {out_bin}')
    print(f'  OpenCV:      {out_cv}')
    print(f'  naive:       {out_naive}')
