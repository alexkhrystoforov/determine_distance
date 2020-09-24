import cv2
import math


input_video = 'IMG_6502.MOV'
cap = cv2.VideoCapture(input_video)

current_frame = 0

ret, frame = cap.read()

while ret:
    cv2.imwrite('frames/frame' + str(current_frame) + '.jpg', frame)
    current_frame += 1

    ret, frame = cap.read()

frameRate = cap.get(5)  # frame rate

while cap.isOpened():
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if not ret:
        break
    if frameId // math.ceil(frameRate):
        filename = "frames_for_calibrating/frame_" + str(int(frameId)) + ".png"
        cv2.imwrite(filename, frame)
cap.release()
