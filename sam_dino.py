from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from rembg.bg import remove

u4 
def sharpen_image(image):
    # Define a sharpening kernel
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])

    # Apply the sharpening filter using filter2D
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    
    return sharpened_image



def getBoxes(image_path):
    image = Image.open(image_path)
    original_image = cv2.imread(image_path)
    model = YOLO("yolo11x-seg.pt")
    results = model.predict(image, conf=0.2)
    detections = results[0]
    extracted_masks = results[0].masks.data  
    print(f"Extracted masks: {extracted_masks}")
    boxes = detections.boxes
    class_ids = boxes.cls.tolist()  


    print(f"Class IDs: {class_ids}")


    # Set up the plot
    num_objects = len(extracted_masks)
    fig, axs = plt.subplots(1, num_objects, figsize=(15, 5))  
    
    if num_objects == 1:
        axs = [axs]  
    
    for i, mask in enumerate(extracted_masks):
        mask = mask.squeeze(0).cpu().numpy()  
        mask = (mask * 255).astype(np.uint8)  
        # Resize the mask to match the image size (if needed)
        mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        # Apply the mask to the original image to extract the object
        object_extracted = cv2.bitwise_and(original_image, original_image, mask=mask_resized)
 
        object_extracted_rgb = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2RGB)
        object=Image.fromarray(object_extracted_rgb)
        object.show()

        # Convert BGR to RGB for displaying in matplotlib
        

        # # Display the extracted object in the subplot
        # axs[i].imshow(object_extracted_rgb)
        # axs[i].axis('off')  
        # axs[i].set_title(f"Object {i+1} - Class {class_ids[i]}")
        # plt.show()





