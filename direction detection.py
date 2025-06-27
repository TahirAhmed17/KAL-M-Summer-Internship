import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Parameters for ShiTomasi corner detection
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create some random colors for drawing tracks
color = np.random.randint(0, 255, (100, 3))

# Initialize variables for tracking
points_to_track = []
old_gray = None
mask = None
directions = []

def get_direction(dx, dy):
    angle = np.degrees(np.arctan2(dy, dx))
    if -45 <= angle <= 45:
        return 'Right'
    elif 45 < angle <= 135:
        return 'Up'
    elif -135 <= angle < -45:
        return 'Down'
    else:
        return 'Left'

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if old_gray is None:
        # Initialize on the first frame
        old_gray = frame_gray.copy()
        # Detect initial good features to track
        points_to_track = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)  # Initialize mask with zeros
    
    else:
        # Calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points_to_track, None, **lk_params)
        
        # Keep only the points that are successfully tracked
        good_new = p1[st == 1]
        good_old = points_to_track[st == 1]
        
        # Update points to track with the newly found good points
        points_to_track = good_new.reshape(-1, 1, 2)
        
        # Draw tracks and points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            # Calculate direction
            dx, dy = a - c, b - d
            direction = get_direction(dx, dy)
            directions.append(direction)
            
            # Display direction
            cv.putText(frame, direction, (int(a), int(b)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    
    # Overlay the tracks on the frame
    img = cv.add(frame, mask)
    
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Update the previous frame
    old_gray = frame_gray.copy()

    # Check if points to track are empty or need to be re-initialized
    if len(points_to_track) < 10:  # Adjust the number of points dynamically as needed
        points_to_track = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)  # Re-initialize mask

cv.destroyAllWindows()
cap.release()
