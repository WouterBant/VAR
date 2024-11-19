import cv2
import numpy as np
import os


class RobotDetection:
    def __init__(self, config):
        self.counter = 0
        self.config = config
        self.path_for_results = "/home/student/Desktop/VAR/assets/robot_detection_vids"
        os.makedirs(self.path_for_results, exist_ok=True)

    def detect(
        self, frame, show_line=True, show_contour_area=True, show_live_detection=True
    ):
        if self.config.get("save_robot_detection_images") and self.counter % 10 == 0:
            cv2.imwrite(f"{self.path_for_results}/img_{self.counter}.jpg", frame)
            self.counter += 1
        image = frame

        image_copy = image.copy()
        height, width = image_copy.shape[:2]
        percentile_horizontal = 0.55
        percentile_to_keep_vertical = 0.95
        # Set the top half of the image to white
        y_position = int(percentile_horizontal * height)
        # right and left side
        x_position = int(
            percentile_to_keep_vertical * width
        )  # Calculate horizontal pixel position
        x_position_2 = int((1 - percentile_to_keep_vertical) * width)

        image[:y_position, :] = [255, 255, 255]
        image[:, :x_position_2, :] = [255, 255, 255]
        image[:, x_position:, :] = [255, 255, 255]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        image_hsv = cv2.cvtColor(
            image_rgb, cv2.COLOR_BGR2HSV
        )  # Convert to HSV for segmentation

        # Define the black color range in HSV
        lower_black = np.array([0, 0, 0])  # Start from very low values
        # upper_black = np.array([180, 255, 100]) # Limit value to capture darker areas
        upper_black = np.array([180, 255, 85])  # Limit value to capture darker areas

        # Create a binary mask where black colors are white and the rest are black
        mask = cv2.inRange(image_hsv, lower_black, upper_black)
        # mask = cv2.bitwise_not(mask)

        # Apply some morphological operations to clean up the mask
        # Erode to remove small black artifacts/ connect larger objects with dilate
        mask = cv2.erode(mask, np.ones((23, 23), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        # Optionally, smooth the edges of the mask/ DO NOT THINK THIS HELPS
        # mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours of the segmented regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a red line for which part we actually use for segmenting/detection
        if show_line:
            image_copy = self.draw_lines(
                image_copy, percentile_horizontal, percentile_to_keep_vertical
            )
            mask = self.draw_lines(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                percentile_horizontal,
                percentile_to_keep_vertical,
            )

        # Makes sure the object has a certain area so we filter out smaller detected black objects
        COUNTOUR_AREA_THRESHOLD = 2000
        BOTTOM_LEFT_OBJECT_HEIGHT_THRESHOLD = 360
        draw_circle_bool = True

        in_dangers = []
        approx_distances = []
        # Draw bounding boxes around detected ducks on the original image
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # Filter small contours (noise)
            if contour_area > COUNTOUR_AREA_THRESHOLD:  # Adjust the threshold if needed
                x, y, w, h = cv2.boundingRect(contour)
                # Probably useful if the bounding boxes start at the bottom of the image ->
                vertical_height_bottom_left_corner = y + h

                # Make sure we only look at bounding boxes that have their bottom left corner lower than 300
                # vertical pixel -> background left bottom corner is typically at higher vertical view
                if (
                    vertical_height_bottom_left_corner
                    > BOTTOM_LEFT_OBJECT_HEIGHT_THRESHOLD
                ):
                    # print(f"left bottom corner is at height: {vertical_height_bottom_left_corner}")
                    # print(f"vertical image height in pixels: {height}")
                    if draw_circle_bool:
                        self.draw_circle(
                            image_copy, (x, vertical_height_bottom_left_corner)
                        )
                    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if show_contour_area:
                        text_position = (
                            x + 10,
                            y + 30,
                        )  # Adjust position for readability
                        cv2.putText(
                            image_copy,
                            f"Area: {int(contour_area)} Width: {w} Height: {h}",
                            text_position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )
                    in_danger, approx_distance = self.determine_distance_danger(h)
                    in_dangers.append(in_danger)
                    approx_distances.append(approx_distance)

                if show_live_detection:
                    cv2.imshow("Robot Detection", image_copy)
                    cv2.waitKey(1)  # Ensure the window updates properly
                    cv2.imshow("Mask Robot", mask)
                    cv2.waitKey(1)  # Ensure the window updates properly

        return in_dangers, approx_distances


def determine_distance_danger(self, bounding_box_height):
    PIXEL_HEIGHT_BOUNDING_BOX_2_CM = 110
    approx_distance = 2 * PIXEL_HEIGHT_BOUNDING_BOX_2_CM / bounding_box_height
    # We need to find correct thresholding values for this
    in_danger = True if approx_distance < 2 else False
    # print(f"Robot is in danger: {in_danger}")
    if in_danger:
        print(f"approximate distance of object: {approx_distance}")
        print("Robot is in danger of collision, we should stop and turn")

    return in_danger, approx_distance

    def draw_lines(self, img, percentile_horizontal, percentile_to_keep_vertical):
        height, width = img.shape[:2]

        y_position = int(percentile_horizontal * height)

        x_position = int(
            percentile_to_keep_vertical * width
        )  # Calculate horizontal pixel position
        x_position_2 = int(
            (1 - percentile_to_keep_vertical) * width
        )  # Calculate horizontal pixel position other side

        # Horizontal Line
        img[y_position - 2 : y_position, x_position_2:x_position, :] = [
            0,
            0,
            255,
        ]  # Red color in BGR format

        # Vertical Lines
        img[y_position:, x_position : x_position + 2, :] = [0, 0, 255]
        img[y_position:, x_position_2 - 2 : x_position_2, :] = [0, 0, 255]
        return img

    def draw_circle(self, img, coordinates):
        radius = 2  # Radius of the circle

        # Define the color (in BGR format)
        color = (0, 0, 255)  # Red color (BGR format)

        # Define the thickness (use -1 for a filled circle)
        thickness = 3  # Outline thickness

        # Draw the circle on the image
        cv2.circle(img, coordinates, radius, color, thickness)
