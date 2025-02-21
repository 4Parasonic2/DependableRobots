import cv2
import numpy as np

def main():
    # Open the default camera (adjust the index if needed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally, resize the frame for faster processing
        # frame = cv2.resize(frame, (600, 400))

        # Blur the frame to reduce high frequency noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Define the HSV ranges for red.
        # Red spans from 0-10 and 170-180 in hue.
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Perform a series of erosions and dilations to remove any small blobs left in the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        # Only proceed if at least one contour was found
        if contours:
            # Find the largest contour in the mask, then use it to compute the minimum enclosing circle
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # Calculate the centroid of the contour
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Only proceed if the radius meets a minimum size
            if radius > 10:
                # Draw the circle around the ball
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # Draw a small circle at the center
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, "Red Ball Detected", (int(x - radius), int(y - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Red Ball Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
