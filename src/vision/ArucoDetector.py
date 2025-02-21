import cv2
import cv2.aruco as aruco

def main():
    # Load the predefined dictionary (here we use 6x6 markers with 250 possible markers)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Create detector parameters using the constructor (no DetectorParameters_create in 4.8)
    detector_params = aruco.DetectorParameters()
    
    # Create the detector object (new in OpenCV 4.8)
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to grab frame from camera.")
            break

        # Convert the frame to grayscale (required for detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers in the grayscale image using the new ArucoDetector
        corners, ids, rejected = detector.detectMarkers(gray)

        # If markers are detected, draw them on the original frame
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Display the resulting frame
        cv2.imshow("Aruco Markers", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



