import tkinter as tk
from PIL import Image, ImageTk
from app.camera import Camera
from app.face_detector import FaceDetector
from app.prediction import Prediction
from app.preprocessing import preprocess_face
import cv2


root = tk.Tk()
camera = Camera()
face_detector = FaceDetector()
prediction = Prediction()
current_frame = None
current_face = None

root.title("Age & Gender Prediction App")
root.configure(bg='#D8D6FF')

menu = tk.Menu(root)
root.config(menu=menu)

filemenu = tk.Menu(menu)
menu.add_cascade(label="Instructions", menu=filemenu)
filemenu.add_command(label="""
            * Upload a picture from your computer or capture a snapshot from your camera
            * Click the 'Run prediction' button to see your predicted age and gender
            * Click the 'Save snapshot' button to save pictures to your own device
             """)

label = tk.Label(root, text="Age & Gender Prediction App", bg="lightgray", fg="black")
label.pack(pady=40)

label1 = tk.Label(root, text="Welcome!", bg="lightgray", fg="black")
label1.pack()

label_img = tk.Label(master=root)
label_img.configure(bg='#D8D6FF')
label_img.pack(pady=40)
label_img.place(relx=0.5, rely=0.5, relwidth=0.6, relheight=0.6, anchor = 'e')

label_result = tk.Label(master=root)
label_result.configure(bg='white')
label_result.pack(pady=40)
label_result.place(relx=0.5, rely=0.5, relwidth=0.6, relheight=0.6, anchor = 'w')

camera.start_camera()

def update_camera():
    global current_frame, current_face
    try:
        frame = camera.read_frame()
        current_frame = frame.copy()
        faces = face_detector.detect_face(frame)
        current_face = faces
        for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (820, 500))
        img = Image.fromarray(resized_img)
        imgtk = ImageTk.PhotoImage(image=img)
        label_img.imgtk = imgtk
        label_img.configure(image=imgtk)
    except Exception as e:
        print("Camera error:", e)
    root.after(250, update_camera) #TODO change update time (250 ms)


def get_prediction():
    largest_face = face_detector.select_largest_face(current_face, current_frame)
 
    if largest_face is None:
        label_result.config(text="No face detected")
        return
    
    result = prediction.predict(largest_face)
    label_result.config(text=result)

button = tk.Button(root, text="Get Prediction", command=get_prediction, activebackground="darkviolet")
button.pack(pady=20)




def close():
    camera.close_camera()
    root.destroy()

update_camera()

quit_button = tk.Button(root, text="Quit Age & Gender Prediction App", command=close)
quit_button.pack(pady=20)


root.protocol("WM_DELETE_WINDOW", close)

root.mainloop()