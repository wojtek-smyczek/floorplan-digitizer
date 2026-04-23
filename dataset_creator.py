import cv2
import os
import numpy as np

def create_dataset_from_sketches(input_folder='sketches', output_folder='my_dataset'):
    """
    Tnie duże arkusze ze szkicami cyfr na pojedyncze obrazki 64x64.
    Obsługuje wiele rzędów i kolumn na jednym zdjęciu.
    """

    def segment_digits_2d(thresh_roi):
        row_sums = cv2.reduce(thresh_roi, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
        rows = []
        in_row = False
        start_y = 0

        for y, val in enumerate(row_sums):
            if val > 50 and not in_row:
                in_row = True
                start_y = y
            elif val <= 50 and in_row:
                in_row = False
                rows.append(thresh_roi[start_y:y, :])

        digits_images = []
        for row in rows:
            column_sums = cv2.reduce(row, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]
            in_digit = False
            start_x = 0

            for x, val in enumerate(column_sums):
                if val > 50 and not in_digit:
                    in_digit = True
                    start_x = x
                elif val <= 50 and in_digit:
                    in_digit = False
                    digit_crop = row[:, start_x:x]

                    if digit_crop.shape[0] > 5 and digit_crop.shape[1] > 5:
                        h, w = digit_crop.shape
                        side = max(h, w) + 40
                        square = np.zeros((side, side), dtype=np.uint8)

                        y_off = (side - h) // 2
                        x_off = (side - w) // 2
                        square[y_off:y_off+h, x_off:x_off+w] = digit_crop

                        final_img = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
                        digits_images.append(final_img)
        return digits_images

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for digit in range(10):
        filename = f"{digit}.jpg"
        img_path = os.path.join(input_folder, filename)

        if not os.path.exists(img_path):
            print(f"[-] Pominąłem: {filename} (nie znaleziono w folderze '{input_folder}')")
            continue

        print(f"[*] Przetwarzam arkusz dla cyfry {digit}...")
        img = cv2.imread(img_path)

        red_channel = img[:, :, 2]
        _, thresh = cv2.threshold(red_channel, 60, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        samples = segment_digits_2d(thresh)

        class_dir = os.path.join(output_folder, str(digit))
        os.makedirs(class_dir, exist_ok=True)

        for i, sample in enumerate(samples):
            cv2.imwrite(os.path.join(class_dir, f"{digit}_sample_{i}.png"), sample)

        print(f"[+] Sukces: Wygenerowano {len(samples)} obrazków dla cyfry {digit}.")

if __name__ == "__main__":
    create_dataset_from_sketches(input_folder='sketches', output_folder='my_dataset')
