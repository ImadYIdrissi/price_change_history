import cv2
import pytesseract

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class TextElement:
    def __init__(self, contour, offset=(0, 0)):
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        self.x += offset[0]
        self.y += offset[1]

    def _preprocess_image(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to remove noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Use adaptive thresholding instead of global thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        # Apply a morphological operation (dilation followed by erosion,
        # known as closing)
        # This can help close small holes or gaps within text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

        return morph

    def ocr_text(self, original_img):
        # Crop the relevant area from the provided original image on-the-fly
        cropped_img = original_img[
            self.y : self.y + self.h, self.x : self.x + self.w
        ]
        # Preprocess the cropped image
        preprocessed_img = self._preprocess_image(cropped_img)
        # Apply pytesseract OCR on the preprocessed image
        text = pytesseract.image_to_string(
            preprocessed_img, config="--psm 7 --oem 3"
        )  # psm 7 is for treating the image as a single text line
        return {
            "cropped_img": preprocessed_img,
            "text": text.strip(),
        }


class Line(TextElement):
    def __init__(self, contour, line_id, offset=(0, 0)):
        super().__init__(contour, offset)
        self.line_id = line_id
        self.phrases = []


class Phrase(TextElement):
    def __init__(self, contour, line_ref, offset=(0, 0)):
        super().__init__(contour, offset)
        self.line_ref = line_ref
        self.words = []


class Word(TextElement):
    def __init__(self, contour, phrase_ref, offset=(0, 0)):
        super().__init__(contour, offset)
        self.phrase_ref = phrase_ref


def preprocess_image_for_contours(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("The image could not be loaded.")
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


def detect_text_elements(img, thresh):
    def _get_vertical_position(contour):
        _, y, _, _ = cv2.boundingRect(contour)
        return y

    def _get_horizontal_position(contour):
        x, _, _, _ = cv2.boundingRect(contour)
        return x

    lines = []
    line_contours = find_contours(thresh, (70, 5), 5)

    # Sort line contours by their vertical (y) position
    sorted_line_contours = sorted(
        line_contours,
        key=_get_vertical_position,
    )

    for line_id, contour in enumerate(sorted_line_contours):
        line = Line(contour, line_id)
        lines.append(line)

        line_thresh = thresh[
            line.y : line.y + line.h, line.x : line.x + line.w
        ]

        phrase_contours = find_contours(line_thresh, (20, 5), 2)
        # Sort phrase contours by their horizontal (x) position
        sorted_phrase_contours = sorted(
            phrase_contours,
            key=_get_horizontal_position,
        )

        for phrase_contour in sorted_phrase_contours:
            phrase = Phrase(
                phrase_contour,
                line.line_id,
                offset=(line.x, line.y),
            )
            line.phrases.append(phrase)

            phrase_thresh = thresh[
                phrase.y : phrase.y + phrase.h, phrase.x : phrase.x + phrase.w
            ]
            word_contours = find_contours(phrase_thresh, (4, 5), 2)
            # Sort word contours by their horizontal (x) position
            sorted_word_contours = sorted(
                word_contours,
                key=_get_horizontal_position,
            )
            # Optionally sort words if needed, similar to phrases
            for word_contour in sorted_word_contours:
                word = Word(word_contour, phrase, offset=(phrase.x, phrase.y))
                phrase.words.append(word)

    return lines


def display_img(mat, tag=""):
    # Show image
    cv2.imshow("Image" + tag, mat)
    # Wait for the user to press a key
    cv2.waitKey(0)
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = (
        "/home/iyid/workspaces/price_change_history/"
        "images/runes/20240309_runes_1.png"
    )
    original_img, thresh = preprocess_image_for_contours(image_path)
    lines = detect_text_elements(original_img, thresh)

    selected_phrases = []
    for line in lines:
        # line.draw(img=original_img, color=BLUE, display=True)
        for i, phrase in enumerate(line.phrases):
            if i != 1:  # Skip the third phrase
                # phrase.draw(img=original_img, color=RED, display=True)
                selected_phrases.append(phrase)
                for word in phrase.words:
                    word_img = word.ocr_text(original_img=original_img)[
                        "cropped_img"
                    ]
                    word_txt = word.ocr_text(original_img=original_img)["text"]
                    print(word_txt)
                    display_img(word_img)
