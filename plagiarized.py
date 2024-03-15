# https://nanonets.com/blog/ocr-with-tesseract/

import cv2
import pytesseract
from pytesseract import Output

img = cv2.imread("images/runes/20240309_runes_1.png")

d = pytesseract.image_to_data(img, output_type=Output.DICT)

n_boxes = len(d["text"])
for i in range(n_boxes):
    if int(d["conf"][i]) > 60:
        (x, y, w, h) = (
            d["left"][i],
            d["top"][i],
            d["width"][i],
            d["height"][i],
        )
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
print(d.keys())
