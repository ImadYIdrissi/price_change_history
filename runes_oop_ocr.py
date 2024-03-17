import cv2
import numpy as np
import pytesseract
from typing import Tuple, List, Dict, Any

# Define color constants for drawing on images
BLUE: Tuple[int, int, int] = (255, 0, 0)
GREEN: Tuple[int, int, int] = (0, 255, 0)
RED: Tuple[int, int, int] = (0, 0, 255)


class TextElement:
    def __init__(
        self, contour: np.ndarray, offset: Tuple[int, int] = (0, 0)
    ) -> None:
        """
        Initialize a TextElement object with a contour and an optional offset.

        Args:
            contour: A numpy array containing contour points of the text
            element.
            offset: A tuple representing the offset to be applied to the
            contour's position, default is (0, 0).
        """
        self.x, self.y, self.w, self.h = cv2.boundingRect(array=contour)
        self.x += offset[0]
        self.y += offset[1]

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for OCR by converting it to grayscale, applying
        Gaussian blur, adaptive thresholding, and a morphological close
        operation.

        Args:
            img: The image to preprocess.

        Returns:
            The preprocessed image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to remove noise
        blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
        # Use adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            src=blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        # Apply a morphological operation
        # - dilation followed by erosion, known as closing
        # - This can help close small holes or gaps within text characters

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2, 2))
        morph = cv2.morphologyEx(
            src=adaptive_thresh, op=cv2.MORPH_CLOSE, kernel=kernel
        )
        return morph

    def ocr_text(self, original_img: np.ndarray) -> Dict[str, Any]:
        """
        Perform OCR on a cropped and preprocessed part of the original image.

        Args:
            original_img: The original image to perform OCR on.

        Returns:
            A dictionary with keys 'cropped_img' and 'text', containing the
            preprocessed image and the OCR-extracted text, respectively.
        """
        cropped_img = original_img[
            self.y : self.y + self.h, self.x : self.x + self.w
        ]
        # Preprocess the cropped image
        preprocessed_img = self._preprocess_image(img=cropped_img)
        # Apply pytesseract OCR on the preprocessed image
        text = pytesseract.image_to_string(
            image=preprocessed_img, config="--psm 7 --oem 3"
        ).strip()
        return {"cropped_img": preprocessed_img, "text": text}


class Line(TextElement):
    def __init__(
        self,
        contour: np.ndarray,
        line_id: int,
        offset: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Initialize a Line object, inheriting from TextElement, to represent a
        line of text.

        Args:
            contour: A numpy array containing contour points of the line.
            line_id: An integer representing the unique identifier of the line.
            offset: A tuple representing the offset to be applied to the line's
                    position, default is (0, 0).
        """
        super().__init__(contour=contour, offset=offset)
        self.line_id = line_id
        self.phrases: List[Phrase] = []


class Phrase(TextElement):
    def __init__(
        self,
        contour: np.ndarray,
        line_ref: int,
        offset: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Initialize a Phrase object, inheriting from TextElement, to represent
        a phrase within a line.

        Args:
            contour: A numpy array containing contour points of the phrase.
            line_ref: An integer reference to the parent Line object.
            offset: A tuple representing the offset to be applied to the
                    phrase's position, default is (0, 0).
        """
        super().__init__(contour=contour, offset=offset)
        self.line_ref = line_ref
        self.words: List[Word] = []


class Word(TextElement):
    def __init__(
        self,
        contour: np.ndarray,
        phrase_ref: Phrase,
        offset: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Initialize a Word object, inheriting from TextElement, to represent a
        word within a phrase.

        Args:
            contour: A numpy array containing contour points of the word.
            phrase_ref: A reference to the parent Phrase object.
            offset: A tuple representing the offset to be applied to the word's
                    position, default is (0, 0).
        """
        super().__init__(contour=contour, offset=offset)
        self.phrase_ref = phrase_ref


def preprocess_image_for_contours(
    img_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image from the given path and preprocess it to find contours.

    Args:
        img_path: The file path of the image.

    Returns:
        A tuple containing the original image and the thresholded image.
    """
    img = cv2.imread(filename=img_path)
    if img is None:
        raise ValueError("The image could not be loaded.")
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        src=gray, thresh=150, maxval=255, type=cv2.THRESH_BINARY
    )
    return img, thresh


def find_contours(
    thresh: np.ndarray, kernel_size: Tuple[int, int], iterations: int
) -> List[np.ndarray]:
    """
    Find contours in the given thresholded image using a specified kernel size
    and number of iterations for dilation.

    Args:
        thresh: The thresholded image.
        kernel_size: A tuple specifying the size of the kernel used for
        dilation.
        iterations: The number of times dilation is applied.

    Returns:
        A list of contours found in the image.
    """
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=kernel_size)
    dilate = cv2.dilate(src=thresh, kernel=kernel, iterations=iterations)
    contours, _ = cv2.findContours(
        image=dilate, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def detect_text_elements(img: np.ndarray, thresh: np.ndarray) -> List[Line]:
    """
    Detect and organize text elements (lines, phrases, words) in the given
    image based on the provided thresholded image.

    Args:
        img: The original image.
        thresh: The thresholded image.

    Returns:
        A list of Line objects, each containing their respective Phrase and
        Word objects.
    """

    def _get_vertical_position(contour: np.ndarray) -> int:
        _, y, _, _ = cv2.boundingRect(array=contour)
        return y

    def _get_horizontal_position(contour: np.ndarray) -> int:
        x, _, _, _ = cv2.boundingRect(array=contour)
        return x

    lines: List[Line] = []
    line_contours = find_contours(
        thresh=thresh, kernel_size=(70, 5), iterations=5
    )

    sorted_line_contours = sorted(line_contours, key=_get_vertical_position)

    for line_id, contour in enumerate(sorted_line_contours):
        line = Line(contour=contour, line_id=line_id)
        lines.append(line)

        line_thresh = thresh[
            line.y : line.y + line.h, line.x : line.x + line.w
        ]
        phrase_contours = find_contours(
            thresh=line_thresh, kernel_size=(20, 5), iterations=2
        )
        sorted_phrase_contours = sorted(
            phrase_contours, key=_get_horizontal_position
        )

        for phrase_contour in sorted_phrase_contours:
            phrase = Phrase(
                contour=phrase_contour,
                line_ref=line.line_id,
                offset=(line.x, line.y),
            )
            line.phrases.append(phrase)

            phrase_thresh = thresh[
                phrase.y : phrase.y + phrase.h, phrase.x : phrase.x + phrase.w
            ]
            word_contours = find_contours(
                thresh=phrase_thresh, kernel_size=(4, 5), iterations=2
            )
            sorted_word_contours = sorted(
                word_contours, key=_get_horizontal_position
            )

            for word_contour in sorted_word_contours:
                word = Word(
                    contour=word_contour,
                    phrase_ref=phrase,
                    offset=(phrase.x, phrase.y),
                )
                phrase.words.append(word)

    return lines


def display_img(mat: np.ndarray, tag: str = "") -> None:
    """
    Display an image with a given tag in the window title.

    Args:
        mat: The image matrix to display.
        tag: An optional tag to append to the window title.
    """
    cv2.imshow(winname="Image" + tag, mat=mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = (
        "/home/iyid/workspaces/price_change_history/"
        "images/runes/20240309_runes_1.png"
    )
    original_img, thresh = preprocess_image_for_contours(img_path=image_path)
    lines = detect_text_elements(img=original_img, thresh=thresh)

    selected_phrases = []
    for line in lines:
        for i, phrase in enumerate(line.phrases):
            if i != 1:  # Skip this phrase
                selected_phrases.append(phrase)
                for word in phrase.words:
                    word_img, word_txt = word.ocr_text(
                        original_img=original_img
                    ).values()
                    print(word_txt)
                    display_img(mat=word_img)
