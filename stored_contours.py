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


def store_element_data(
    element_list,
    parent_id,
    contour,
    prefix,
    offset=(0, 0),
):
    x, y, w, h = cv2.boundingRect(contour)
    x_offset, y_offset = offset
    element_data = {
        "id": len(element_list),
        # "id": f"{prefix}_{len(element_list) + 1}",
        "x": x + x_offset,
        "y": y + y_offset,
        "w": w,
        "h": h,
    }
    if parent_id:
        element_data["belongs_to"] = parent_id
    element_list.append(element_data)
    return element_data["id"]


def detect_and_store_elements(img_path):
    original_img, thresh = preprocess_image(img_path)

    list_of_lines = []
    list_of_phrases = []
    list_of_words = []

    # Detect Lines
    line_contours = find_contours(thresh, (70, 5), 5)
    for line_contour in line_contours:
        line_id = store_element_data(list_of_lines, None, line_contour, "line")

        x, y, w, h = cv2.boundingRect(line_contour)
        line_thresh = thresh[y : y + h, x : x + w]

        # Detect Phrases within Line
        phrase_contours = find_contours(line_thresh, (20, 5), 2)
        for phrase_contour in phrase_contours:
            phrase_id = store_element_data(
                list_of_phrases,
                line_id,
                phrase_contour,
                "phrase",
                offset=(x, y),
            )

            px, py, pw, ph = cv2.boundingRect(phrase_contour)
            phrase_thresh = line_thresh[py : py + ph, px : px + pw]

            # Detect Words within Phrase
            word_contours = find_contours(phrase_thresh, (5, 5), 2)
            for word_contour in word_contours:
                store_element_data(
                    list_of_words,
                    phrase_id,
                    word_contour,
                    "word",
                    offset=(x + px, y + py),
                )

    return list_of_lines, list_of_phrases, list_of_words


def plot_bounding_boxes(img, elements_list, color):
    """
    Draws bounding boxes on the image based on the provided list of
      dictionaries.
    """
    for data in elements_list:
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    image_path = (
        "/home/iyid/workspaces/price_change_history/"
        "images/runes/20240309_runes_1.png"
    )
    original_img = cv2.imread(image_path)

    lines, phrases, words = detect_and_store_elements(image_path)

    # Example: Plot lines in red
    plot_bounding_boxes(original_img.copy(), lines, (0, 0, 255))
