from .marker_detection import MarkerDetection
from .localization import Localization
from .robot_detection import RobotDetection


class PipeLine:
    def __init__(self, config):
        self.config = config
        self.marker_detector = MarkerDetection(config=self.config)
        self.localization = Localization(config=self.config)
        self.robot_detector = RobotDetection(config=self.config)
        # self.action_controller = ActionController(config=self.config)

    def run(self, cv_image):
        if self.config.get("detect_marker"):
            marker_detection_results = self.marker_detector.detect(cv_image)
            location = self.localization.triangulate(marker_detection_results)
            if self.config.get("debug") > 2:
                print(f"Location: {location}")

        if self.config.get("detect_robot"):
            img, mask = self.robot_detector.detect(cv_image)
            if self.config.get("debug") > 2:
                print(f"Robot detection: {img}, {mask}")

        # inDanger, distances = False, None
        # if self.config.get("avoid_obstacles"):
        #     inDanger, distances = self.object_detector.detect(cv_image)
        # action = self.action_controller.plan(location, inDanger, distances)
        action = (0, 0)
        return action

    def __call__(self, cv_image):
        return self.run(cv_image)
