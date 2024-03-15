from glob import glob
from pathlib import Path
from datetime import datetime
from IPython.display import display
from PIL import Image  # , ImageFilter, ImageOps

import cv2
import numpy as np
import pytesseract


def image_to_str(image):
    """Return a string from an image using OCR."""
    return pytesseract.image_to_string(image)


def convert_to_grayscale_pil(img):
    """Function to convert to grayscale using PIL."""
    return img.convert("L")


def convert_to_grayscale_cv2(img):
    """Function to convert to grayscale using OpenCV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_otsu_threshold(gray_img):
    """Function to apply Otsu's binarization using OpenCV."""
    return cv2.threshold(
        gray_img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]


def dilate_image_cv2(img):
    """Function to dilate the image using OpenCV."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(img, kernel, iterations=1)


def erode_image_cv2(img):
    """Function to erode the image using OpenCV."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(img, kernel, iterations=1)


def save_image_cv2(img, path):
    """Saves the OpenCV image to the specified path."""
    cv2.imwrite(str(path), img)


def process_and_save_strips_cv2(
    image,
    header_height,
    strip_height,
    base_path,
    ext="png",
    after_header_offset=0,
):
    """Process the OpenCV image into strips and save each one."""
    strips = []
    height, width = image.shape[:2]
    current_height = header_height + after_header_offset

    # Save the header separately if required
    if header_height > 0:
        header = image[:header_height, 0:width]
        header_path = base_path / f"header.{ext}"
        save_image_cv2(header, header_path)

    # Process and save the remaining strips
    i = 1
    while current_height < height:
        strip = image[
            current_height : min(
                current_height + strip_height,
                height,
            ),
            0:width,
        ]
        strip_path = base_path / f"strip_{i}.{ext}"
        save_image_cv2(strip, strip_path)
        strips.append(strip)
        current_height += strip_height
        i += 1

    return strips


def process_and_save_strips_pil(
    image,
    header_height,
    strip_height,
    base_path,
    ext="png",
    after_header_offset=0,
):
    """Process the PIL image into strips and save each one."""
    strips = []
    width, height = image.size
    current_height = header_height + after_header_offset

    # Save the header separately if required
    if header_height > 0:
        header = image.crop((0, 0, width, header_height))
        header_path = base_path / f"header.{ext}"
        header.save(header_path)
        strips.append(header)

    # Process and save the remaining strips
    i = 1
    while current_height < height:
        strip = image.crop(
            (
                0,
                current_height,
                width,
                min(
                    current_height + strip_height,
                    height,
                ),
            )
        )
        strip_path = base_path / f"strip_{i}.{ext}"
        strip.save(strip_path)
        strips.append(strip)
        current_height += strip_height
        i += 1

    return strips


if __name__ == "__main__":
    input_path_prefix = "images/runes/20240309_runes_"
    output_data_path = Path("output/data/runes")
    output_intermediary_images = Path("output/images/runes")
    ext = "png"

    list_with_many_images = glob(pathname=f"{input_path_prefix}*.{ext}")

    for i, image_path in enumerate(list_with_many_images):
        basename = Path(image_path).stem
        ext = Path(image_path).suffix.split(".")[-1]
        date_str = datetime.now().strftime("%Y%m%d")

        image = Image.open(image_path)
        base_output_path = output_intermediary_images / date_str / str(i + 1)

        # Preprocess and save grayscale image
        gray_image_cv2 = convert_to_grayscale_cv2(cv2.imread(image_path))
        save_image_cv2(
            gray_image_cv2,
            base_output_path / "grayscale" / f"{basename}_grayscale.png",
        )

        # Apply Otsu's threshold and save binarized image
        thresh_image = apply_otsu_threshold(gray_image_cv2)
        save_image_cv2(
            thresh_image,
            base_output_path / "binarized" / f"{basename}_binarized.png",
        )

        # Dilate, erode, and save the preprocessed images
        dilated_image = dilate_image_cv2(thresh_image)
        save_image_cv2(
            dilated_image,
            base_output_path / "dilated" / f"{basename}_dilated.png",
        )
        eroded_image = erode_image_cv2(dilated_image)
        save_image_cv2(
            eroded_image,
            base_output_path / "eroded" / f"{basename}_eroded.png",
        )

        # Convert the eroded image back to PIL format for the slicing function
        eroded_image_pil = Image.fromarray(eroded_image)

        final_preprocessed_image = gray_image_cv2

        # Slice the dilated image into strips, save them, and perform OCR
        header_height = 43  # The header height in pixels
        strip_height = 67  # The height of each strip in pixels
        after_header_offset = 5  # The additional offset after the header

        strips_output_dir = base_output_path / "strips"
        strips_output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the strips directory exists
        strips = process_and_save_strips_cv2(
            final_preprocessed_image,
            header_height,
            strip_height,
            strips_output_dir,
            after_header_offset,
        )

        # OCR each strip and write the results to a CSV file
        csv_output_path = output_data_path / str(i + 1)
        csv_output_path.mkdir(parents=True, exist_ok=True)
        csv_output_path = csv_output_path / f"{basename}.csv"
        with open(csv_output_path, "w", encoding="utf-8") as csv_file:
            for strip in strips:
                display(strip)
                text = image_to_str(strip)
                # csv_file.write(text + "\n")
                print()
                print(text)

    print("End")
