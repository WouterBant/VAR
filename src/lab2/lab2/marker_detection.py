import cv2
import apriltag


class MarkerDetection:
    def __init__(self, config):
        self.config = config
        self.detector = apriltag.Detector()

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh)
        results = self.detector.detect(denoised)
        
        if self.config.get("debug") > 0:
            if results:
                for r in results:
                    tag_id = r.tag_id
                    center = tuple(map(int, r.center))
                    cv2.circle(image, center, 5, (0, 255, 0), -1)
                    cv2.putText(image, str(tag_id), center, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.waitKey(1)

        return results