import cv2


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return img, thresh


def find_contours(thresh, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(thresh, kernel, iterations=iterations)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def store_element_data(element_dict, parent_id, contour, prefix, offset=(0, 0)):
    x, y, w, h = cv2.boundingRect(contour)
    x_offset, y_offset = offset
    element_id = f"{prefix}_{len(element_dict) + 1}"
    element_data = {
        "x": x + x_offset,
        "y": y + y_offset,
        "w": w,
        "h": h,
    }
    if parent_id:
        element_data["belongs_to"] = parent_id
    element_dict[element_id] = element_data
    return element_id


def detect_and_store_elements(img_path):
    original_img, thresh = preprocess_image(img_path)

    dict_of_lines = {}
    dict_of_phrases = {}
    dict_of_words = {}

    # Detect Lines
    line_contours = find_contours(thresh, (70, 5), 5)
    for line_contour in line_contours:
        line_id = store_element_data(dict_of_lines, None, line_contour, "line")

        x, y, w, h = cv2.boundingRect(line_contour)
        line_thresh = thresh[y : y + h, x : x + w]

        # Detect Phrases within Line
        phrase_contours = find_contours(line_thresh, (20, 5), 2)
        for phrase_contour in phrase_contours:
            phrase_id = store_element_data(
                dict_of_phrases, line_id, phrase_contour, "phrase", offset=(x, y)
            )

            px, py, pw, ph = cv2.boundingRect(phrase_contour)
            phrase_thresh = line_thresh[py : py + ph, px : px + pw]

            # Detect Words within Phrase
            word_contours = find_contours(phrase_thresh, (5, 5), 2)
            for word_contour in word_contours:
                store_element_data(
                    dict_of_words,
                    phrase_id,
                    word_contour,
                    "word",
                    offset=(x + px, y + py),
                )

    return dict_of_lines, dict_of_phrases, dict_of_words


def plot_bounding_boxes(img, elements_dict, color):
    """
    Draws bounding boxes on the image based on the provided dictionary.

    Parameters:
    - img: The original image on which to draw the bounding boxes.
    - elements_dict: A dictionary containing the bounding box information.
      The expected format is: {"element_id": {"x": int, "y": int, "w": int, "h": int, ...}, ...}
    - color: The color of the bounding boxes (B, G, R) tuple.
    """
    for element_id, data in elements_dict.items():
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Optionally, display the image with bounding boxes
    cv2.imshow("Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_combined_bounding_boxes(img, elements_dicts_colors):
    """
    Draws bounding boxes for multiple element types (lines, phrases, words) on the image.

    Parameters:
    - img: The original image on which to draw the bounding boxes.
    - elements_dicts_colors: A list of tuples, each containing an element dictionary and its corresponding box color.
      Format: [({"element_id": {"x": int, "y": int, "w": int, "h": int, ...}}, (B, G, R)), ...]
    """
    for elements_dict, color in elements_dicts_colors:
        for element_id, data in elements_dict.items():
            x, y, w, h = data["x"], data["y"], data["w"], data["h"]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Optionally, display the image with bounding boxes
    cv2.imshow("Combined Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = (
        "/home/iyid/workspaces/price_change_history/"
        "images/runes/20240309_runes_1.png"
    )
    original_img = cv2.imread(image_path)

    lines, phrases, words = detect_and_store_elements(image_path)

    # Plot lines in red, phrases in green, and words in blue
    # plot_bounding_boxes(original_img.copy(), lines, (0, 0, 255))  # Red for lines
    # plot_bounding_boxes(original_img.copy(), phrases, (0, 255, 0))  # Green for phrases
    # plot_bounding_boxes(original_img.copy(), words, (255, 0, 0))  # Blue for words

    # Plot all elements on the same image with different colors
    elements_dicts_colors = [
        (lines, (0, 0, 255)),  # Red for lines
        (phrases, (0, 255, 0)),  # Green for phrases
        (words, (255, 0, 0)),  # Blue for words
    ]
    plot_combined_bounding_boxes(original_img.copy(), elements_dicts_colors)
