from glob import glob
from pathlib import Path
from datetime import datetime
from IPython.display import display
from PIL import Image, ImageFilter, ImageOps

import pytesseract


def image_to_str(image):
    """Return a string from an image using OCR."""
    return pytesseract.image_to_string(image)


def convert_to_grayscale(img):
    """Function to convert to grayscale."""
    return img.convert("L")


def binarize_image(img, threshold=128):
    """Function to apply binarization."""
    return img.point(lambda p: p > threshold and 255)


def dilate_image(img):
    """Function to dilate the image."""
    return img.filter(ImageFilter.MaxFilter(size=3))


def invert_colors(img):
    """Function to invert the colors of the image."""
    return ImageOps.invert(img)


def erode_image(img):
    """Function to erode the image."""
    # First, invert the image if the text is light and the background is dark
    inverted_img = invert_colors(img)
    # Apply erosion to the inverted image
    eroded_img = inverted_img.filter(ImageFilter.MinFilter(3))
    # Re-invert the image to restore the original color scheme
    return invert_colors(eroded_img)


def save_image(img, path):
    """Saves the image to the specified path."""

    if not isinstance(path, Path):
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    img.save(path)


def process_and_save_strips(
    image, header_height, strip_height, base_path, after_header_offset=0
):
    """Process the image into strips and save each one."""
    strips = []
    width, height = image.size
    current_height = header_height + after_header_offset

    # Save the header separately if required
    if header_height > 0:
        header = image.crop((0, 0, width, header_height))
        header_path = base_path / f"header.{ext}"
        save_image(header, header_path)
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
        save_image(strip, strip_path)
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
        gray_image = convert_to_grayscale(image)
        save_image(
            gray_image, base_output_path / "grayscale" / f"{basename}_grayscale.{ext}"
        )

        # Preprocess and save binarized image
        binarized_image = binarize_image(gray_image)
        save_image(
            binarized_image,
            base_output_path / "binarized" / f"{basename}_binarized.{ext}",
        )

        # Preprocess and save dilated image
        dilated_image = dilate_image(binarized_image)
        save_image(
            dilated_image, base_output_path / "dilated" / f"{basename}_dilated.{ext}"
        )

        # Preprocess and save eroded image
        eroded_image = dilate_image(dilated_image)
        save_image(
            eroded_image, base_output_path / "eroded" / f"{basename}_eroded.{ext}"
        )

        final_preprocessed_image = gray_image

        # Slice the dilated image into strips, save them, and perform OCR
        header_height = 43  # The header height in pixels
        strip_height = 67  # The height of each strip in pixels
        after_header_offset = 5  # The additional offset after the header

        strips_output_dir = base_output_path / "strips"
        strips_output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the strips directory exists
        strips = process_and_save_strips(
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
