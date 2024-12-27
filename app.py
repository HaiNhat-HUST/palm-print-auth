import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import random

# ----------------------
# Global Variables
# ----------------------
original_img = None   # Will store the original loaded image
hog_img = None        # This should be assigned your HoG image after processing
roi_img = None        # This should be assigned your ROI image after processing
displayed_original = None  # Used for the label
displayed_processed = None # Used for the label (either HoG or ROI)
hello_world_label = None   # For printing "hello world" or any future prediction text

def choose_file():
    global original_img, lbl_left_img

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    if not file_path:
        return
    original_img = Image.open(file_path)

    # ---------------------------
    # Resize the image to 400x400
    # If you want original resolution,
    # remove or adjust the resizing:
    # ---------------------------
    original_img = original_img.resize((400, 400))
    tk_img = ImageTk.PhotoImage(original_img)

    # Update the label for the original image
    lbl_left_img.config(image=tk_img)
    lbl_left_img.image = tk_img  # Keep a reference so it's not GC'd

def show_roi():
    global roi_img, lbl_right_img

    if not original_img:
        return

    # ---------------------------
    # In practice, you would generate your HoG image here.
    # For demonstration, let's just reuse the original image.
    # Replace the below lines with your actual HoG processing logic.
    # ---------------------------
    roi_img = original_img  # your_hog_function(original_img)

    tk_hog = ImageTk.PhotoImage(roi_img)
    lbl_right_img.config(image=tk_hog)
    lbl_right_img.image = tk_hog

def show_hog():
    global hog_img, lbl_right_img

    if not original_img:
        return  # If no image is loaded, do nothing

    # ---------------------------
    # In practice, you would generate your HoG image here.
    # For demonstration, let's just reuse the original image.
    # Replace the below lines with your actual HoG processing logic.
    # ---------------------------
    hog_img = original_img  # your_hog_function(original_img)

    # Convert hog_img to ImageTk
    tk_hog = ImageTk.PhotoImage(hog_img)
    lbl_right_img.config(image=tk_hog)
    lbl_right_img.image = tk_hog

def predict():
    global entry_username

    thr_min = 30
    thr_max = 70
    prediction_value = random.randint(0, 100)

    if prediction_value < thr_min:
        messagebox.showinfo("Prediction", "Logged in successfully!")
    elif thr_min <= prediction_value <= thr_max:
        messagebox.showinfo("Prediction", "Please enter your password.")
        entry_username.config(state='normal')
    else:
        messagebox.showerror("Prediction", "Your palm is not in our database.")

def login():
    if entry_username.cget('state') == 'disabled':
        messagebox.showerror("Error", "Please perform prediction first.")
        return

    password = entry_username.get()
    correct_password = "password123"

    if password == correct_password:
        messagebox.showinfo("Login", "Logged in successfully!")
    else:
        messagebox.showerror("Login", "Incorrect password. Please try again.")

def main():
    global original_image, processed_image, lbl_left_img, lbl_right_img, entry_username

    root = tk.Tk()
    root.title("Biometrics Demonstration App")

    # ----------------------------
    # Top Frame: two sub-frames (left & right) with an arrow in between
    # ----------------------------
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    # Left Frame
    original_image = tk.Frame(top_frame)
    original_image.pack(side=tk.LEFT, padx=10)

    lbl_left_img = tk.Label(original_image, text="Original image")
    lbl_left_img.pack()

    # Arrow in the middle
    arrow_label = tk.Label(top_frame, text="→", font=("Arial", 24))
    arrow_label.pack(side=tk.LEFT, padx=10)

    # Right Frame
    processed_image = tk.Frame(top_frame)
    processed_image.pack(side=tk.LEFT, padx=10)

    lbl_right_img = tk.Label(processed_image, text="Processed image")
    lbl_right_img.pack()

    # ----------------------------
    # Button to choose image (below the two frames)
    # ----------------------------
    btn_choose = tk.Button(root, text="Choose Image", command=choose_file)
    btn_choose.pack(pady=5)

    # ----------------------------
    # Frame for ROI, HOG, Predict buttons
    # ----------------------------
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)

    btn_roi = tk.Button(btn_frame, text="ROI", command=show_roi)
    btn_roi.pack(side=tk.LEFT, padx=5)

    btn_hog = tk.Button(btn_frame, text="HOG", command=show_hog)
    btn_hog.pack(side=tk.LEFT, padx=5)

    btn_predict = tk.Button(btn_frame, text="Predict", command=predict)
    btn_predict.pack(side=tk.LEFT, padx=5)

    # ----------------------------
    # Frame for text field and Login button
    # ----------------------------
    login_frame = tk.Frame(root)
    login_frame.pack(pady=10)

    entry_username = tk.Entry(login_frame, width=20, state='disabled')
    entry_username.pack(side=tk.LEFT, padx=5)

    btn_login = tk.Button(login_frame, text="Login", command=login)
    btn_login.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
