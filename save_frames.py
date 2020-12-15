import os
import cv2

os.chdir(r'C:\Users\jwmcc\Python\speedchallenge\data')

for cat in ['train', 'test']:
    cap = cv2.VideoCapture(f'{cat}.mp4')
    while cap.isOpened():
        # read current frame
        frame_id = int(cap.get(1))
        ret, frame = cap.read()
        if not ret:
            break

        # save frame
        filename = f'{cat}/{cat}_{frame_id}.jpg'
        if not cv2.imwrite(filename, frame):
            print('Failed to write frame ' + frame_id)

    cap.release()
    print(f'Finished saving {cat} data')