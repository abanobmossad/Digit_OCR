import cv2
import craft_utils
import imgproc
from craft import CRAFT
from torch.autograd import Variable
from collections import OrderedDict
import torch
import numpy as np

def craft_text_detection(image_path, output_path):
    # Load the pre-trained CRAFT model
    net = CRAFT()  # You can pass in other parameters for the model if needed
    net.load_state_dict(
        craft_utils.copyStateDict(torch.load('craft_weights/craft_mlt_25k.pth')))
    net = net.cuda()

    # Set the model to evaluation mode
    net.eval()

    # Read the input image
    image = cv2.imread(image_path)

    # Resize the image to the size expected by the model
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image)

    # Normalize the image
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    x = Variable(x.cuda(), volatile=True)

    # Forward pass
    y, _ = net(x)

    # Post-process the output
    boxes, _ = craft_utils.getDetBoxes(y[0, :, :, :].detach().cpu().numpy(), target_ratio, size_heatmap)

    # Draw bounding boxes on the original image
    for i in range(len(boxes)):
        boxes[i] = boxes[i] * target_ratio
    img_copied = image.copy()
    for box in boxes:
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(img_copied, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)

    # Save the result
    cv2.imwrite(output_path, img_copied)

if __name__ == '__main__':
    # Specify the path to your local image
    input_image_path = 'test_images/Trials.jpg'

    # Specify the path for saving the output image with bounding boxes
    output_image_path = 'test_images/Segmented image'

    # Perform text detection and save the result
    craft_text_detection(input_image_path, output_image_path)
