from .marker_detection import MarkerDetection
from .localization import Localization


class PipeLine:
    def __init__(self, config):
        self.config = config
        self.marker_detector = MarkerDetection(config=self.config)
        self.localization = Localization(config=self.config)
        # self.object_detector = ObjectDetection(config=self.config)
        # self.action_controller = ActionController(config=self.config)

    def run(self, cv_image):
        marker_detection_results = self.marker_detector.detect(cv_image)
        location = self.localization.triangulate(marker_detection_results)
        inDanger, distances = False, None
        # if self.config.get("avoid_obstacles"):
        #     inDanger, distances = self.object_detector.detect(cv_image)
        # action = self.action_controller.plan(location, inDanger, distances)
        # return action

    def __call__(self, cv_image):
        return self.run(cv_image)
