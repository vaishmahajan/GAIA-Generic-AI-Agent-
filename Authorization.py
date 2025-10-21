import cv2

# Load the CascadeClassifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Create a counter for the detected faces
face_counter = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the original frame
        face = frame[y:y+h, x:x+w]
        
        # Save the face to a file
        cv2.imwrite(f'detected_face_{face_counter}.jpg', face)
        
        # Increment the face counter
        face_counter += 1
        
        # Draw a rectangle around the face in the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the output
    cv2.imshow('Live Face Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()