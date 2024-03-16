import cv2


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return img, thresh


def find_contours(thresh, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(thresh, kernel, iterations=iterations)
    contours, _ = cv2.findContours(
        dilate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def highlight_elements(img, contours, offset=(0, 0), color=(0, 255, 0)):
    x_offset, y_offset = offset
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            img,
            (x + x_offset, y + y_offset),
            (x + x_offset + w, y + y_offset + h),
            color,
            2,
        )
    return img


def detect_and_highlight(img_path):
    original_img, thresh = preprocess_image(img_path)

    # Detect Lines
    line_contours = find_contours(thresh, (70, 5), 5)
    original_img = highlight_elements(
        original_img,
        line_contours,
        color=(255, 0, 0),
    )

    # For each line, detect phrases
    for line_contour in line_contours:
        x, y, w, h = cv2.boundingRect(line_contour)
        line_thresh = thresh[y : y + h, x : x + w]
        phrase_contours = find_contours(line_thresh, (10, 5), 2)
        original_img = highlight_elements(
            original_img,
            phrase_contours,
            offset=(x, y),
            color=(0, 255, 0),
        )

        # For each phrase, detect words
        for phrase_contour in phrase_contours:
            px, py, pw, ph = cv2.boundingRect(phrase_contour)
            phrase_thresh = line_thresh[py : py + ph, px : px + pw]
            word_contours = find_contours(phrase_thresh, (5, 5), 2)
            original_img = highlight_elements(
                original_img,
                word_contours,
                offset=(x + px, y + py),
                color=(0, 0, 255),
            )

    # Show the result
    cv2.imshow("Highlighted Text", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = (
        "/home/iyid/workspaces/price_change_history/"
        "images/runes/20240309_runes_1.png"
    )
    detect_and_highlight(image_path)
