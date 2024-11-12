import cv2


ARUCO_DICT = {
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
  "DICT_ARUCO_MIP_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12
}

frame = cv2.imread("image3.jpeg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame = cv2.adaptiveThreshold(
#             frame, 255, 
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2
#         )

# crop the image
frame = frame[500:-700, 1050:-700]

all_corners, all_marker_ids = [], []
for desired_aruco_dictionary in ARUCO_DICT.keys():
  # Load the desired ArUco dictionary
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters()
    # import code; code.interact(local=dict(globals(), **locals()))
    # 
    this_aruco_parameters.adaptiveThreshWinSizeMin = 3
    this_aruco_parameters.adaptiveThreshWinSizeMin = 5
    this_aruco_parameters.adaptiveThreshWinSizeMax = 21
    this_aruco_parameters.adaptiveThreshWinSizeStep = 4
    this_aruco_parameters.minMarkerPerimeterRate = 0.07
    # this_aruco_parameters.minMarkerPerimeterRate = 0.01
    # this_aruco_parameters.maxMarkerPerimeterRate = 4.0

    (corners, ids, rejected) = cv2.aruco.detectMarkers(  # TODO maybe use rejected
        frame, this_aruco_dictionary, parameters=this_aruco_parameters)
    all_corners.append(corners)
    all_marker_ids.append(ids)
        
# Check that at least one ArUco marker was detected
if len(corners) > 0:
    # Flatten the ArUco IDs list
    ids = ids.flatten()
    
    # Loop over the detected ArUco corners
    for (marker_corner, marker_id) in zip(corners, ids):
    
        # Extract the marker corners
        corners = marker_corner.reshape((4, 2))
        (top_left, top_right, bottom_right, bottom_left) = corners
        
        # Convert the (x,y) coordinate pairs to integers
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))
        
        # Draw the bounding box of the ArUco detection
        cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
        
        # Calculate and draw the center of the ArUco marker
        center_x = int((top_left[0] + bottom_right[0]) / 2.0)
        center_y = int((top_left[1] + bottom_right[1]) / 2.0)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Draw the ArUco marker ID on the video frame
        # The ID is always located at the top_left of the ArUco marker
        cv2.putText(frame, str(marker_id), 
        (top_left[0], top_left[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 2)

cv2.imshow('frame', frame)
try:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    pass

if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()