import math
import cv2
import numpy as np
from time import time

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(1)

# Initialize a variable to store the time of the previous frame.
prev_time = time()

# Initialize a counter for screenshot naming
screenshot_count = 1

# Iterate until the video is accessed successfully.
while video.isOpened():
    
    # Read a frame.
    ok, frame = video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Break the loop.
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the current time.
    current_time = time()
    
    # Check if 10 seconds have passed since the last screenshot.
    if current_time - prev_time >= 10:
        # Save the current frame as a screenshot
        cv2.imwrite(f'screenshot_{screenshot_count}.png', frame)
        screenshot_count += 1
        # Update the previous time
        prev_time = current_time
    
    # Display the frame.
    cv2.imshow('Live Video', frame)
    
    # Wait until a key is pressed.
    # Retrieve the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        # Break the loop.
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()
