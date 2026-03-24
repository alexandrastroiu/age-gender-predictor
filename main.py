import cv2
import os
import datetime
from app.camera import Camera
from app.face_detector import FaceDetector

def save_image(face):
    directory_name = "data/faces"

    try:
        os.makedirs(directory_name, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while saving image: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/snapshots/face_{timestamp}.jpg"
    cv2.imwrite(filename, face)
    print(f"Saved face snapshot to: {filename}")
        

def main():
    camera = Camera()
    detector = FaceDetector()

    try:
        camera.start_camera()

        while True:
            frame = camera.read_frame()
            faces = detector.detect_face(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            largest_face = FaceDetector.select_largest_face(faces, frame)
            
            cv2.imshow("Face Detection", frame)    

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                if largest_face is not None:
                    save_image(largest_face)
                else:
                    print("No face detected")
            

    finally:
        camera.close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()