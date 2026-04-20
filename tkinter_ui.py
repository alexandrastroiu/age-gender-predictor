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

# Se foloseste biblioteca tkinter pentru a crea interfata grafica a aplicatiei desktop
# Creeaz fereastra aplicatiei
root = tk.Tk()
# Creeaza cate un obiect, instantiaza Camera, Face Detector si Prediction
camera = Camera()
face_detector = FaceDetector()
prediction = Prediction()
current_frame = None
current_face = None

# Titlul aplicatiei
root.title("Age & Gender Prediction App")
root.configure(bg="#EAE8F8")

font = tkinter.font.Font(
    family="Helvetica",
    size=14,
)
font_header = tkinter.font.Font(family="Helvetica", size=16, weight="bold")
font_title = tkinter.font.Font(family="Helvetica", size=32, weight="bold")

menu = tk.Menu(root)
root.config(menu=menu)

# Meniu de instructiuni
filemenu = tk.Menu(menu)
menu.add_cascade(label="Instructions", menu=filemenu, font=font)
filemenu.add_command(
    label="""
            * Upload a picture from your computer or capture a snapshot from your camera
            * Click the 'Run prediction' button to see your predicted age and gender
            * Click the 'Share results' button to share results online
             """,
    font=font,
)

label = tk.Label(
    root,
    text="Age & Gender Prediction App \U0001f52e",
    bg="#EAE8F8",
    fg="#2C2C2C",
    font=font_title,
)
label.pack(pady=40)


label1 = tk.Label(root, text="Welcome!", bg="#EAE8F8", fg="#2C2C2C", font=font_header)
label1.pack()

separator = tk.Frame(root, height=2, bg="#2C2C2C")
separator.pack(fill="x", pady=10)

# Label pentru imaginea de la webcam
label_img = tk.Label(master=root)
label_img.configure(bg="#EAE8F8")
label_img.pack(pady=40)
label_img.place(relx=0.5, rely=0.5, relwidth=0.6, relheight=0.6, anchor="e")

# Label pentru rezultatele predictiei
label_result = tk.Label(
    master=root, text="Prediction Results", font=font_header, anchor="n"
)
label_result.configure(bg="white")
label_result.pack(pady=40)
label_result.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.53, anchor="w")

# Labels pentru fiecare rezultat al predictiei
box_gender = tk.Label(
    label_result,
    text="Gender: ",
    width=50,
    height=2,
    relief="solid",
    bd=1,
    anchor="center",
    font=font,
    bg="white",
)
box_gender.grid(row=0, column=0, padx=50, pady=50)

box_gender_conf = tk.Label(
    label_result,
    text="Gender Prediction Accuracy: ",
    width=50,
    height=2,
    relief="solid",
    bd=1,
    anchor="center",
    font=font,
    bg="white",
)
box_gender_conf.grid(row=1, column=0, padx=50, pady=25)

box_age = tk.Label(
    label_result,
    text="Age Range: ",
    width=50,
    height=2,
    relief="solid",
    bd=1,
    anchor="center",
    font=font,
    bg="white",
)
box_age.grid(row=2, column=0, padx=50, pady=25)

box_age_conf = tk.Label(
    label_result,
    text="Age Prediction Accuracy: ",
    width=50,
    height=2,
    relief="solid",
    bd=1,
    anchor="center",
    font=font,
    bg="white",
)
box_age_conf.grid(row=3, column=0, padx=50, pady=25)

# Porneste camera
camera.start_camera()

# Actualizeaza interfata grafica pentru a afisa cadrele de la camera
def update_camera():
    global current_frame, current_face
    try:
        # Citeste un cadru de la camera si il returneaza
        frame = camera.read_frame()
        # Pastreaza o copie a cadrului curent de la camera
        current_frame = frame.copy()
        # Detecteaza fata din imagine
        faces = face_detector.detect_face(frame)
        # Pastreaza o copie a fetei detectate in cadrul curent de la camera
        current_face = faces
        # Deseneaza un dreptunghi pentru a evidentia fata detectata in cadru
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Face conversia de la formatul BGR (formatul default al OpenCV) la RGB deoarece va fi folosita biblioteca PIL
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Redimensioneaza imaginea pentru afisarea in interfata grafica
        resized_img = cv2.resize(img, (850, 500))
        img = Image.fromarray(resized_img)
        # Face conversia imaginii intr-un format pe care Tkinter poate sa il afiseze
        imgtk = ImageTk.PhotoImage(image=img)
        # Pastreaza o referinta la imagine
        label_img.imgtk = imgtk
        # Asociaza label-ul cu noua imagine care va fi afisata in interfata grafica
        label_img.configure(image=imgtk)
    except Exception as e:
        print("Camera error:", e)
    # Actualizeaza imaginea din interfata grafica apeland functia la cate 250 ms
    root.after(250, update_camera)


# Functia va realiza afisarea rezulatelor predictiei in interfata grafica
def get_prediction():
    # Selecteaza cea mai mare fata detectata in cadrul de la camera
    largest_face = face_detector.select_largest_face(current_face, current_frame)

    # In cazul in care nicio fata nu este detectata in cadru, nu se fac predictii
    if largest_face is None:
        label_result.config(text="No face detected")
        box_gender.config(text=f"Gender: - ")
        box_gender_conf.config(text=f"Gender Prediction Accuracy: - %")
        box_age.config(text=f"Age Range: - ")
        box_age_conf.config(text=f"Age Prediction Accuracy: - %")
        return
    else:
        if label_result.cget("text") != "Prediction Results":
            label_result.config(text="Prediction Results")

    # Se afiseaza rezultatele predictiei in interfata grafica: genul, acuratetea predictiei pentru gen, categoria de varsta, acuratetea predictiei pentru categoria de varsta
    result = prediction.predict(largest_face)
    box_gender.config(text=f"Gender: {result["Gender"]}")
    box_gender_conf.config(
        text=f"Gender Prediction Accuracy: {(result["Gender Confidence"] * 100):.2f} %"
    )
    box_age.config(text=f"Age Range: {result["Age Category"]}")
    box_age_conf.config(
        text=f"Age Prediction Accuracy: {result["Age Confidence"] * 100 :.2f} %"
    )


# Distribuie rezultatele predictiei folosind aplicatia de mail a utilizatorului
# Modulul urllib din Python este folosit pentru a lucra cu URL-uri si a face cereri HTTP
def share_results():
    result_text = (
        f"{box_gender.cget('text')}\n"
        f"{box_gender_conf.cget('text')}\n"
        f"{box_age.cget('text')}\n"
        f"{box_age_conf.cget('text')}"
    )

    subject = urllib.parse.quote("My Age & Gender Prediction App Results")
    body = urllib.parse.quote(
        f"Hi!\n\nSee my results on the Age & Gender Prediction App:\n\n{result_text}"
    )

    # Modulul webbrowser din Python este folosit in general pentru a deschide documente web in browser-ul default al utilizatorului
    # Este folosit un link mailto pentru a deschide aplicatia default de mail a utilizatorului, subiectul si corpului email-ului sunt deja completate
    webbrowser.open(f"mailto:?subject={subject}&body={body}")


# Inchide aplicatia
def close_app():
    # Elibereaza resursele
    camera.close_camera()
    # Distruge toate obiectele de tip widget din fereastra si iese din mainloop, inchide aplicatia
    root.destroy()


# Actualizeaza interfata de la webcam in interfata grafica
update_camera()

# Butoane pentru: quit, share, realizarea predictiei
quit_button = tk.Button(
    root,
    text="Quit Age & Gender Prediction App",
    command=close_app, # Functia ce va fi apelata la apasarea butonului
    font=font_header,
    width=30,
    height=1,
)
quit_button.pack(side="bottom", pady=10)

share_button = tk.Button(
    root,
    text="Share Results",
    font=font_header,
    width=30,
    height=1,
    command=share_results,
)
share_button.pack(side="bottom", pady=10)

prediction_button = tk.Button(
    root,
    text="Get Prediction",
    command=get_prediction,
    font=font_header,
    width=30,
    height=1,
)
prediction_button.pack(side="bottom", pady=10)

# Se foloseste metoda protocol() pentru a apela functia "close app" atunci cand utilizatorul apasa butonul de inchidere din bara ferestrei
root.protocol("WM_DELETE_WINDOW", close_app)

# La pornirea aplicatiei intra in mainloop
# In mainloop pastreaza fereastra deschisa, asculta evenimente (tastatura, click-uri) si actualizeaza interfata grafica continuu
root.mainloop()
