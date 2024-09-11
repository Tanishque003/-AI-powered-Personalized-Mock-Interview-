import cv2

# Open default camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture")
else:
    print("Camera opened successfully")

# Release the camera
cap.release()
cv2.destroyAllWindows()
