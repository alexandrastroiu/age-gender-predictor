import tkinter as tk
import tkinter.font
from PIL import Image, ImageTk
from app.camera import Camera
from app.face_detector import FaceDetector
from app.prediction import Prediction
from app.preprocessing import preprocess_face
import cv2
import webbrowser
import urllib.parse


root = tk.Tk()
camera = Camera()
face_detector = FaceDetector()
prediction = Prediction()
current_frame = None
current_face = None

root.title("Age & Gender Prediction App")
root.configure(bg="#EAE8F8")

font = tkinter.font.Font(family="Helvetica", size=14,)
font_header = tkinter.font.Font(family="Helvetica", size=16, weight="bold")
font_title = tkinter.font.Font(family="Helvetica", size=32, weight="bold")

menu = tk.Menu(root)
root.config(menu=menu)

filemenu = tk.Menu(menu)
menu.add_cascade(label="Instructions", menu=filemenu, font=font)
filemenu.add_command(label="""
            * Upload a picture from your computer or capture a snapshot from your camera
            * Click the 'Run prediction' button to see your predicted age and gender
            * Click the 'Share results' button to hare results online
             """, font=font)

label = tk.Label(root, text="Age & Gender Prediction App", bg="#EAE8F8",fg="#2C2C2C", font=font_title)
label.pack(pady=40)


label1 = tk.Label(root, text="Welcome!", bg="#EAE8F8",fg="#2C2C2C", font=font_header)
label1.pack()

separator = tk.Frame(root, height=2, bg="#2C2C2C")
separator.pack(fill="x", pady=10)

label_img = tk.Label(master=root)
label_img.configure(bg='#EAE8F8')
label_img.pack(pady=40)
label_img.place(relx=0.5, rely=0.5, relwidth=0.6, relheight=0.6, anchor = 'e')

label_result = tk.Label(master=root, text="Prediction Results", font=font_header, anchor="n")
label_result.configure(bg='white')
label_result.pack(pady=40)
label_result.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.53, anchor = 'w')

box_gender = tk.Label(label_result, text="Gender: ", width=50, height=2, relief="solid", bd=1, anchor="center", font=font, bg="white")
box_gender.grid(row=0, column=0, padx=50, pady=50)

box_gender_conf = tk.Label(label_result, text="Gender Prediction Confidence: ", width=50, height=2, relief="solid", bd=1, anchor="center",font=font, bg="white")
box_gender_conf.grid(row=1, column=0, padx=50, pady=25)

box_age = tk.Label(label_result, text="Age Range: ", width=50, height=2, relief="solid", bd=1, anchor="center", font=font, bg="white")
box_age.grid(row=2, column=0, padx=50, pady=25)

box_age_conf = tk.Label(label_result, text="Age Prediction Confidence: ", width=50, height=2, relief="solid", bd=1, anchor="center", font=font, bg="white")
box_age_conf.grid(row=3, column=0, padx=50, pady=25)


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
        resized_img = cv2.resize(img, (850, 500))
        img = Image.fromarray(resized_img)
        imgtk = ImageTk.PhotoImage(image=img)
        label_img.imgtk = imgtk
        label_img.configure(image=imgtk)
    except Exception as e:
        print("Camera error:", e)
    root.after(250, update_camera)


def get_prediction():
    largest_face = face_detector.select_largest_face(current_face, current_frame)
 
    if largest_face is None:
        label_result.config(text="No face detected")
        return
    
    result = prediction.predict(largest_face)
    box_gender.config(text=f"Gender: {result["Gender"]}")
    box_gender_conf.config(text=f"Gender Prediction Confidence: {(result["Gender Confidence"] * 100):.2f} %")
    box_age.config(text=f"Age Range: {result["Age Category"]}")
    box_age_conf.config(text=f"Age Prediction Confidence: {result["Age Confidence"] * 100 :.2f} %")


def share_results():
    result_text = (
        f"{box_gender.cget('text')}\n"
        f"{box_gender_conf.cget('text')}\n"
        f"{box_age.cget('text')}\n"
        f"{box_age_conf.cget('text')}"
    )

    subject = urllib.parse.quote("My Age & Gender Prediction App Results")
    body = urllib.parse.quote(f"Hi!\n\nSee my results on the Age & Gender Prediction App:\n\n{result_text}")

    webbrowser.open(f"mailto:?subject={subject}&body={body}")
        


def close():
    camera.close_camera()
    root.destroy()

update_camera()


quit_button = tk.Button(root, text="Quit Age & Gender Prediction App", command=close, font=font_header, width=30, height=1)
quit_button.pack(side="bottom", pady=10)

share_button = tk.Button(root, text="Share Results", font=font_header, width=30, height=1, command=share_results)
share_button.pack(side="bottom", pady=10)

button = tk.Button(root, text="Get Prediction", command=get_prediction, font=font_header, width=30, height=1)
button.pack(side="bottom", pady=10)


root.protocol("WM_DELETE_WINDOW", close)

root.mainloop()