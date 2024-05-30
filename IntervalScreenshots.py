import cv2
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import SendApiReq as API
import threading

API_KEY = "//////////"

INTERVAL_TIME = 3

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils 

# Function to detect pose
def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(1)

# Create named window for resizing purposes
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# Initialize the VideoCapture object to read from a video stored in the disk.
#video = cv2.VideoCapture('media/running.mp4')

# Set video camera size
video.set(3,1280)
video.set(4,960)

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Initialize a variable to store the time of the last screenshot.
last_screenshot_time = time()

# Create directory to save screenshots
os.makedirs("screenshots", exist_ok=True)

# Counter to keep track of screenshot index
screenshot_counter = 0

# Thread function to create API call
def print_api_output(prompt, screenshot_filename, API_KEY):
    response = API.get_api_req(prompt, screenshot_filename, API_KEY)
    print(f'{screenshot_filename}: {response}%')

# Iterate until the video is accessed successfully.
while video.isOpened():
    
    # Read a frame.
    ok, org_frame = video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Break the loop.
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    org_frame = cv2.flip(org_frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  org_frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    org_frame = cv2.resize(org_frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(org_frame, pose_video, display=False)
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
    if (time2 - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2
    
    


    # Check if n seconds have elapsed since the last screenshot
    if time2 - last_screenshot_time >= INTERVAL_TIME:
        # Save the screenshot
        screenshot_filename = f"screenshots/screenshot_{screenshot_counter}.png"
        cv2.imwrite(screenshot_filename, org_frame)
        # TODO: SCREENSHOT SAVED AS
        # print(f"Screenshot saved as {screenshot_filename}")
        
        # Update last_screenshot_time and screenshot_counter
        last_screenshot_time = time2
        screenshot_counter += 1

        # If people are detected
        if landmarks:
            # API Request
            prompt = 'What is the danger rating in this image from 1 to 100? Only respond with a single integer output from 1 to 100. Danger is defined as someone who is posing a threat or looks to be able to cause harm.'
            api_thread = threading.Thread(target=print_api_output, args=(prompt, screenshot_filename, API_KEY))
            api_thread.start()
    
    
    # Display the frame.
    cv2.imshow('Pose Detection', frame)
    
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