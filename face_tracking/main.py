import os
import cv2
import variables as vars
from KF import KalmanFilter


fourcc = cv2.VideoWriter_fourcc(*'XVID')

roi = None
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
kalman = KalmanFilter()


def get_roi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        vars.start_pos = [x, y]
        vars.select_done = False
    elif event == cv2.EVENT_LBUTTONUP:
        if vars.start_pos[0] == x and vars.start_pos[1] == y:
            vars.select_done = False
        else:
            vars.end_pos = [x, y]
            vars.select_done = True


if __name__ == "__main__":
    file_name = "fuyou_1a.mp4"
    file_name2 = "fuyou_1a.avi"
    input_file = os.path.join("../origin_videos/unnormal/", file_name)
    output_file = os.path.join("../processed_videos/unnormal/", file_name2)

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.namedWindow('source')
    cv2.setMouseCallback('source', get_roi)
    out = cv2.VideoWriter(output_file, fourcc, 20, (1920, 1080))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if vars.select_done:
            roi = gray[vars.start_pos[1]:vars.end_pos[1], vars.start_pos[0]:vars.end_pos[0]]
            faces = face_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(120, 120))

            if len(faces) == 0:
                pass

            for (x, y, w, h) in faces:
                # img = cv2.rectangle(frame,(x+vars.start_pos[0],y+vars.start_pos[1]),(x+vars.start_pos[0]+w,y+vars.start_pos[1]+h),(255,0,0),2)
                p_x = x
                p_y = y
                if vars.last_face_x == 0 and vars.last_face_y == 0:
                    vars.last_face_x = p_x
                    vars.last_face_y = p_y
                else:
                    real_dev_x = p_x - vars.last_face_x
                    real_dev_y = p_y - vars.last_face_y
                    dev_x, dev_y = kalman.predict(real_dev_x, real_dev_y)
                    vars.start_pos[0] += dev_x
                    vars.start_pos[1] += dev_y
                    vars.end_pos[0] += dev_x
                    vars.end_pos[1] += dev_y

                    if vars.start_pos[0] >= frame.shape[1]:
                        vars.start_pos[0] = frame.shape[1] - 1
                    elif vars.start_pos[0] < 0:
                        vars.start_pos[0] = 0

                    if vars.start_pos[1] >= frame.shape[0]:
                        vars.start_pos[1] = frame.shape[0] - 1
                    elif vars.start_pos[1] < 0:
                        vars.start_pos[1] = 0

                    if vars.end_pos[0] >= frame.shape[1]:
                        vars.end_pos[0] = frame.shape[1] - 1
                    elif vars.end_pos[0] < 0:
                        vars.end_pos[0] = 0

                    if vars.end_pos[1] >= frame.shape[0]:
                        vars.end_pos[1] = frame.shape[0] - 1
                    elif vars.end_pos[1] < 0:
                        vars.end_pos[1] = 0

            frame[0:vars.start_pos[1], :] = 0
            frame[vars.end_pos[1]:1080, :] = 0
            frame[:, 0:vars.start_pos[0]] = 0
            frame[:, vars.end_pos[0]:1920] = 0

            # center_x = (vars.end_pos[0]-vars.start_pos[0]) // 2
            # center_y = (vars.end_pos[1] - vars.start_pos[1]) // 2
            # print(center_y)

            # save_stream =frame[center_y-240:center_y+240,center_x-320:center_x+320]
            # print(save_stream.shape)
            out.write(frame)

        cv2.imshow('source', frame)

        if roi is not None:
            cv2.imshow('roi', roi)

        if cv2.waitKey(1000 / 30) == ord("q"):
            break

    cap.release()
    out.release()
