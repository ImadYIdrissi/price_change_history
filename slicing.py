import cv2
import pytesseract

# import numpy as np

# from PIL import Image


def display_img(mat, tag=""):
    # Show image
    cv2.imshow("Image" + tag, mat)
    # Wait for the user to press a key
    cv2.waitKey(0)
    # Close all windows
    cv2.destroyAllWindows()


def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold
    _, binary_img = cv2.threshold(
        src=gray,
        thresh=150,
        maxval=255,
        type=cv2.THRESH_BINARY,  # To make text white, better for dilation
        # type=cv2.THRESH_BINARY_INV,  # To make text black
    )

    return binary_img


def find_text_lines(thresh):
    # Define a kernel for morphological operation
    # debug_show_img(mat=thresh, tag="thresh")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 5))

    # Dilate to connect text lines
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    # debug_show_img(mat=dilate, tag="dilate")
    # Find contours
    contours, _ = cv2.findContours(
        dilate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def draw_and_show_contours(img, contours):

    # Convert binary image to BGR for visualization
    if len(img.shape) == 2:  # Check if the image is grayscale (binary)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    # Draw contours on the image
    cv2.drawContours(
        image=img_color,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=1,
    )  # Draw in green with a thin line
    # Show the image with contours
    cv2.imshow("Contours on Image", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def slice_lines(img, contours):
    lines = []
    for contour in contours:
        # Get the rectangle bounding the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Slice the image using the bounds
        line = img[y : y + h, x : x + w]
        lines.append(line)

    dict_lines = {i: line for i, line in enumerate(lines)}
    return dict_lines


def slice_phrases(row_index: int, binary_img_line):
    # Similar process as find_text_lines but adjust kernel size
    # and dilate iterations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilate = cv2.dilate(binary_img_line, kernel, iterations=2)

    # display_img(mat=dilate, tag="dilate word")

    contours, _ = cv2.findContours(
        dilate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # draw_and_show_contours(img=binary_img_line, contours=contours)

    phrases = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        phrase = binary_img_line[y : y + h, x : x + w]
        # display_img(mat=phrase, tag="phrase")
        phrases.append(phrase)

    return {"row_id": row_index, "images_of_phrases": phrases}


def slice_words(row_index: int, phrase_index: int, binary_img):
    # Similar process as find_text_lines but adjust kernel size
    # and dilate iterations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(binary_img, kernel, iterations=2)

    # display_img(mat=dilate, tag="dilate word")

    contours, _ = cv2.findContours(
        dilate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # draw_and_show_contours(img=binary_img_line, contours=contours)

    words = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        word = binary_img[y : y + h, x : x + w]

        # display_img(mat=word, tag="word")
        words.append(word)

    return {
        "row_id": row_index,
        "phrase_id": phrase_index,
        "images_of_words": words,
    }


def apply_ocr(image):
    # TODO : Does not work well with all chars... enhance preprocessing?

    # Rescale the image, Tesseract recommends at least 300 DPI for text
    scale_factor = 3
    image = cv2.resize(
        src=image,
        dsize=None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # # Convert to binary (not needed if your image is already binary)
    # if len(image_large.shape) == 2:  # Check if the image is grayscale,binary
    #     _, image_binary = cv2.threshold(
    #         image_large,
    #         0,
    #         255,
    #         cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    #     )
    # else:
    #     image_binary = image_large.copy()

    # get grayscale image
    # image = get_grayscale(image)
    # image = remove_noise(image)

    # Appy dilation*
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_dilated = cv2.dilate(image, kernel, iterations=2)

    # Apply OCR with PSM 10 which assumes a single character is present
    # custom_config = r"--psm 11"
    custom_config = r"--oem 3 --psm 6"

    display_img(image_dilated)
    text = pytesseract.image_to_string(
        image_dilated,
        config=custom_config,
    )

    return text.strip()


if __name__ == "__main__":
    image_path = (
        "/home/iyid/workspaces/price_change_history/"
        "images/runes/20240309_runes_1.png"
    )
    # Load the original image
    binary_img = cv2.imread(image_path)

    binary_img = preprocess_image(
        img=binary_img,
    )
    text_lines_contours = find_text_lines(thresh=binary_img)

    dict_lines = slice_lines(img=binary_img, contours=text_lines_contours)

    dict_ocrd_words = {}
    dict_ocrd_words_exp = {}
    for i, line_bimg in dict_lines.items():
        #      debug_show_img(mat=img, tag=f"line {i}")
        dict_images_of_phrases = slice_phrases(
            row_index=i,
            binary_img_line=line_bimg,
        )

        for j, phrase_bimg in enumerate(
            dict_images_of_phrases["images_of_phrases"],
        ):
            dict_images_of_words = slice_words(
                row_index=i,
                phrase_index=j,
                binary_img=phrase_bimg,
            )

            for word_bimg in dict_images_of_words["images_of_words"]:
                dict_ocrd_words[i] = {
                    j: {"img": word_bimg, "text": apply_ocr(word_bimg)}
                }

                dict_ocrd_words_exp["row_id"] = i
                dict_ocrd_words_exp["phrase_id"] = j
                dict_ocrd_words_exp["img"] = word_bimg

                # Preprocess bimg
                ocrd_text = apply_ocr(word_bimg)
                dict_ocrd_words_exp["text"] = ocrd_text

    print("End")
