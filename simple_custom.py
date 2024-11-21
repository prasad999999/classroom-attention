import numpy as np
import cv2
from detectfaces import get_faces
from keras.models import load_model
import face_recognition
import time
import pandas as pd
from datetime import datetime


# Current date and time for logging
now = datetime.now()
date = str(now.strftime("%d-%m-%Y %H:%M")).split(' ')[0].replace('-', '/').encode()

# Load known face encodings and names
face_data = [
    ("prasad", "images/prasad.jpg"),
    ("atharva", "images/atharva.jpg"),
    ("vaibhav", "images/vaibhav.jpg"),
    ("mrinmayee", "images/mrinmayee.jpg"),
    ("deepali", "images/deepali.jpg"),
]

known_face_encodings = []
known_face_names = []

for name, path in face_data:
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Initialize student tracking and attendance data
t_students = {name: {'focus': 0, 'distract': 0, 'attendance': 0} for name in known_face_names}
df = pd.read_csv('Custom/Evaluation.csv')

# Get today's date as a string in 'DD/MM/YYYY' format
today_date = datetime.now().strftime('%d/%m/%Y')

# Check if today's date column exists, if not, add the column and mark all students as 'Absent'
if today_date not in df.columns:
    df[today_date] = 'Absent'  # Mark all students as 'Absent'

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
attendance = []

img_rows, img_cols = 48, 48
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (255, 255, 255)
box_color = (255, 245, 152)

# Load models for emotion classification and ensemble
model = []
print('Loading Models...')
for i in range(2):
    m = load_model('saved_model/cnn' + str(i) + '.h5')
    model.append(m)
    print(f'Model {i + 1}/3 loaded')

m = load_model('saved_model/ensemble.h5')
model.append(m)
print('Ensemble model loaded\nLoading complete!')

# Predict function using ensemble
def predict(x):
    x_rev = np.flip(x, 1)  # Reverse the image horizontally
    x = x.astype('float32')
    x_rev = x_rev.astype('float32')
    x /= 255
    x_rev /= 255
    
    p = np.zeros((1, 14))  # Two sets of 7 predictions (for original and flipped image)
    p[:, 0:7] = model[0].predict(x.reshape(1, 48, 48, 1))
    p[:, 7:14] = model[1].predict(x_rev.reshape(1, 48, 48, 1))
    
    pre = model[2].predict(p)
    return pre

# Tracking states of students for focus and distraction time
t_states = {name: {'focus_start': None, 'distract_start': None} for name in known_face_names}

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

# Start of session
start_session_time = time.time()

while True:
    ret, img = cap.read()
    curTime = time.time()

    # Get detected faces
    faces = get_faces(img, method='haar')
    for i, (face, x, y, w, h) in enumerate(faces):
        pre = predict(face)  # Predict the emotion
        emotion_index = np.argmax(pre)
        emotion_label = emotion_labels[emotion_index]
        emotion_confidence = int(pre[0, emotion_index] * 100)

        name = ''
        try:
            # Resize and process frame for face recognition
            small_frame = cv2.resize(img[y-20:y+h+20, x-20:x+w+20], (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # Convert from BGR to RGB

            if process_this_frame:
                # Detect face locations first
                face_locations = face_recognition.face_locations(small_frame)

                # Proceed if faces are detected
                if face_locations:
                    # Get face encodings based on the detected locations
                    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # Check if a match was found
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]
                            t_students[name]['attendance'] = 1  # Mark attendance
                            if name not in attendance:
                                attendance.append(name)

        except IndexError:
            print("Error: Index out of range during face encoding")
        except Exception as e:
            print(f"An error occurred: {str(e)}")




        # Process focus and distraction times
        if name != "Unknown" and name:
            if emotion_label in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']:
                if t_states[name]['focus_start'] is not None:
                    focus_duration = curTime - t_states[name]['focus_start']
                    t_students[name]['focus'] += focus_duration
                    t_states[name]['focus_start'] = None

                if t_states[name]['distract_start'] is None:
                    t_states[name]['distract_start'] = curTime

            else:
                if t_states[name]['distract_start'] is not None:
                    distract_duration = curTime - t_states[name]['distract_start']
                    t_students[name]['distract'] += distract_duration
                    t_states[name]['distract_start'] = None

                if t_states[name]['focus_start'] is None:
                    t_states[name]['focus_start'] = curTime

        # Drawing bounding boxes and emotion info
        tl = (x, y)
        br = (x + w, y + h)
        coords = (x, y - 2)

        # Change box color based on the emotion
        if emotion_label in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']:
            box_color = (0, 0, 255)
            txt = f"{name} {emotion_label} [{emotion_confidence}%] | Distracted"
        else:  # Neutral (Focused)
            box_color = (255, 245, 152)
            txt = f"{name} {emotion_label} [{emotion_confidence}%] | Focused"

        # Draw the box and label on the image
        img = cv2.rectangle(img, tl, br, box_color, 2)
        cv2.putText(img, txt, coords, font, 0.8, text_color, 1, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Camera', img)

    # Check for 'q' key to quit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # End of session processing
        # End of session processing
        end_session_time = time.time()
        total_session_time = end_session_time - start_session_time

        for name in attendance:  # Only process students whose faces were detected
            if name in t_students:
                # Use the actual focus and distraction times in seconds, no normalization
                focus_time = t_students[name]['focus']  # Already in seconds
                distract_time = t_students[name]['distract']  # Already in seconds

                # If the fields contain NaN, initialize them to 0
                if pd.isna(df.loc[df['Name'] == name, 't_focused']).any():
                    df.loc[df['Name'] == name, 't_focused'] = 0.0
                if pd.isna(df.loc[df['Name'] == name, 't_distracted']).any():
                    df.loc[df['Name'] == name, 't_distracted'] = 0.0

                # Update the DataFrame with the actual accumulated times
                df.loc[df['Name'] == name, 't_focused'] += focus_time
                df.loc[df['Name'] == name, 't_distracted'] += distract_time

                # Update total time
                df.loc[df['Name'] == name, 't_total'] = df.loc[df['Name'] == name, 't_focused'] + df.loc[df['Name'] == name, 't_distracted']

                # Mark attendance as Present for today's date
                df.loc[df['Name'] == name, today_date] = 'Present'

                        # After the loop, you can mark absent for students who were not detected
                df.loc[~df['Name'].isin(attendance), today_date] = 'Absent'
        
        # Save the updated CSV with attendance and times
        df.to_csv('Custom/Evaluation.csv', index=False)
        break


cap.release()
cv2.destroyAllWindows()
