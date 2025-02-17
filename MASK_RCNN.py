import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load pre-trained Mask R-CNN model from TensorFlow Hub
mask_rcnn_model = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2/1")

# Load input image
image_path = "text.png"  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image to the model's expected input size
image_resized = cv2.resize(image_rgb, (1024, 1024))

# Normalize image pixels
image_norm = image_resized / 255.0

# Expand dimensions to create batch (batch size = 1)
image_expanded = np.expand_dims(image_norm, axis=0)

# Run inference with the model
result = mask_rcnn_model(image_expanded)

# Extract masks, boxes, and labels
masks = result['detection_masks'][0].numpy()
boxes = result['detection_boxes'][0].numpy()
labels = result['detection_class_entities'][0].numpy()
scores = result['detection_scores'][0].numpy()

# Filter predictions based on score threshold
score_threshold = 0.5
filtered_indices = np.where(scores > score_threshold)[0]

# Draw masks and bounding boxes on the image
for idx in filtered_indices:
    mask = masks[idx]
    box = boxes[idx]
    label = labels[idx]
    score = scores[idx]
    
    # Convert mask to binary
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Get bounding box coordinates
    ymin, xmin, ymax, xmax = box
    ymin, xmin, ymax, xmax = int(ymin * image.shape[0]), int(xmin * image.shape[1]), int(ymax * image.shape[0]), int(xmax * image.shape[1])

    # Draw the mask on the image
    mask_colored = np.zeros_like(image)
    mask_colored[ymin:ymax, xmin:xmax] = np.stack([mask] * 3, axis=-1)  # Add color channels
    image = cv2.addWeighted(image, 1.0, mask_colored, 0.5, 0)

    # Draw the bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label_text = f"{label.decode('utf-8')} ({score:.2f})"
    cv2.putText(image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Show the image with masks and boxes
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
cv2.imshow("Image with Mask R-CNN Segmentation", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite("output_image.jpg", image_bgr)
