import cv2
import numpy as np
import os

def remove_lines(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find horizontal lines using a kernel
    kernel = np.ones((1, 5), np.uint8)
    lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Subtract lines from the original image
    cleaned_image = cv2.subtract(image, cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR))

    return cleaned_image

def segment_and_save_digits(image_path, output_folder):
    # Read the input image
    original_image = cv2.imread(image_path)

    # Remove horizontal lines from each digit
    cleaned_original_image = remove_lines(original_image)

    # Convert the cleaned image to grayscale
    gray_image = cv2.cvtColor(cleaned_original_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the contours
    for idx, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract each digit from the original image
        digit = cleaned_original_image[y:y + h, x:x + w]

        # Save each digit as a separate image
        digit_path = os.path.join(output_folder, f'digit_{idx + 1}.png')
        cv2.imwrite(digit_path, digit)

        # Draw a rectangle around the digit on the original image
        cv2.rectangle(cleaned_original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Segmented Digits', cleaned_original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'cropped_text_85.jpg'
output_folder = 'test_images/Segmented image'
segment_and_save_digits(image_path, output_folder)
