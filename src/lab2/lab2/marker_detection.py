import cv2


class MarkerDetection:
    def __init__(self, config):
        self.config = config
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def detect(self, image):
        print("yo")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        print(ids)
        if self.config.get("debug") > 0:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.imshow("ArUco Markers", image)
            cv2.waitKey(1)

        return ids, corners
