import cv2
import numpy as np

# Load the EAST text detection model using readNetFromTensorflow
net = cv2.dnn.readNetFromTensorflow("frozen_east_text_detection.pb")

# Load the image
image_path = "test_images/IMG_3200jpg.jpg"
image = cv2.imread(image_path)
orig_image = image.copy()

# Get image dimensions
height, width = image.shape[:2]

# Create a blob with the original image size
blob = cv2.dnn.blobFromImage(image, 1.0, (3200,3200), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# Set the blob as input to the network
net.setInput(blob)

# Forward pass to get the output scores and geometry
(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

# Get the bounding boxes and probabilities
rectangles, confidences = [], []
for y in range(0, scores.shape[2]):
    scores_data = scores[0, 0, y]
    x_data0 = geometry[0, 0, y]
    x_data1 = geometry[0, 1, y]
    x_data2 = geometry[0, 2, y]
    x_data3 = geometry[0, 3, y]
    angles_data = geometry[0, 4, y]

    for x in range(0, scores.shape[3]):
        if scores_data[x] < 0.5:
            continue

        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        angle = angles_data[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = x_data0[x] + x_data2[x]
        w = x_data1[x] + x_data3[x]

        endX = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
        endY = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        rectangles.append((startX, startY, endX, endY))
        confidences.append(scores_data[x])

# Apply non-maximum suppression to suppress weak, overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(rectangles, confidences, 0.5, 0.3)

# Draw the bounding boxes on the image
if len(indices) > 0:
    for i in indices.flatten():
        (startX, startY, endX, endY) = rectangles[i]
        cv2.rectangle(orig_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Save the cropped images
for i in indices.flatten():
    (startX, startY, endX, endY) = rectangles[i]
    cropped_text = orig_image[startY:endY, startX:endX]

    # Save the cropped text image
    cv2.imwrite(f"cropped_text_{i}.jpg", cropped_text)

    # Draw the bounding box on the original image
    cv2.rectangle(orig_image, (startX, startY), (endX, endY), (0, 255, 0), 2)


# Display the result
cv2.imshow("Text Detection", orig_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
