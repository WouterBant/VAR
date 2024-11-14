import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


class RobotDetection:
    def __init__(self, config):
        self.counter = 0
        self.config = config 
        self.path_for_results = "/home/student/Desktop/VAR/assets/robot_detection_vids"
        os.makedirs(self.path_for_results, exist_ok=True)

    def detect(self, frame, show_line=True):
        if self.config.get("save_robot_detection_images") and self.counter % 10 == 0:
            cv2.imwrite(f"{self.path_for_results}/img_{self.counter}.jpg", frame)
        self.counter += 1
        image = frame
        image_copy = image.copy()

        height, width = image.shape[:2]
        percentile = 0.65
        # Set the top percentile of the image to white
        image[:int(percentile * height), :] = [255, 255, 255]

        # for i in range(image.shape[0]):  # Loop through the height (rows)
        #     for j in range(image.shape[1]):  # Loop through the width (columns)
        #         # Get the RGB values of the pixel (OpenCV uses BGR by default)
        #         b, g, r = image[i, j]

        #         # Calculate the sum of RGB values
        #         if r + g + b > 90:
        #             # Set the pixel to white (255, 255, 255)
        #             image[i, j] = [255, 255, 255]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for segmentation

        # Define the black color range in HSV
        lower_black = np.array([0, 0, 0])      # Start from very low values
        upper_black = np.array([180, 255, 100]) # Limit value to capture darker areas

        # Create a binary mask where black colors are white and the rest are black
        mask = cv2.inRange(image_hsv, lower_black, upper_black)
        # mask = cv2.bitwise_not(mask)

        # Apply some morphological operations to clean up the mask
        mask = cv2.erode(mask, np.ones((20, 20), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        # Optionally, smooth the edges of the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours of the segmented regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a red line for which part we actually use for segmenting
        if show_line:
            image_copy[int(percentile * height)-2:int(percentile * height), :]  = [0, 0, 255]  # Red color in BGR format
        # Draw bounding boxes around detected ducks on the original image
        for contour in contours:
            # print(cv2.contourArea(contour), " check contour")
            # Filter small contours (noise)
            contour_area = cv2.contourArea(contour)
            if contour_area > 1000:  # Adjust the threshold if needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # print(contour_area, " check contour")
                if self.config.get("show_contour_area"):
                # Add the contour area as text inside the rectangle
                    text_position = (x + 10, y + 30)  # Adjust position for readability
                    cv2.putText(image_copy, f"Area: {int(contour_area)}", text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.imshow("Robot Detection", image_copy)
                cv2.waitKey(1)  # Ensure the window updates properly
                cv2.imshow("Mask Robot", mask)
                cv2.waitKey(1)  # Ensure the window updates properly


        return image_copy, mask
