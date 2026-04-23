import cv2
import numpy as np


def _pad_and_resize(digit_crop):
    """Pad do kwadratu i resize do 64x64."""
    h, w = digit_crop.shape
    side = max(h, w) + 40
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit_crop
    return cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)


def segment_digits(thresh_roi):
    column_sums = cv2.reduce(thresh_roi, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]
    roi_h = thresh_roi.shape[0]

    raw_segments = []
    in_digit = False
    start_col = 0
    for x, val in enumerate(column_sums):
        if val > 0 and not in_digit:
            in_digit = True
            start_col = x
        elif val == 0 and in_digit:
            in_digit = False
            if x - start_col >= 3:
                raw_segments.append((start_col, x))
    if in_digit and thresh_roi.shape[1] - start_col >= 3:
        raw_segments.append((start_col, thresh_roi.shape[1]))

    final_segments = []
    for seg_start, seg_end in raw_segments:
        seg_w = seg_end - seg_start
        if seg_w < roi_h * 0.15:
            continue

        seg_sums = column_sums[seg_start:seg_end]
        max_sum = max(seg_sums)

        if seg_w > roi_h * 0.9:
            margin = int(seg_w * 0.2)
            search_zone = seg_sums[margin:seg_w - margin]
            if len(search_zone) > 0:
                min_idx = int(np.argmin(search_zone)) + margin
                min_val = seg_sums[min_idx]
                if min_val < max_sum * 0.5:
                    split_x = seg_start + min_idx
                    final_segments.append((seg_start, split_x))
                    final_segments.append((split_x, seg_end))
                    continue

        final_segments.append((seg_start, seg_end))

    digits_images = []
    for seg_start, seg_end in final_segments:
        digit_crop = thresh_roi[:, seg_start:seg_end]
        if digit_crop.shape[1] > 2:
            digits_images.append(_pad_and_resize(digit_crop))
    return digits_images
