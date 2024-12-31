import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from proie import PROIE
import cv2
from hog import process_hog
import matplotlib.cm as cm
from calc_threshold import find_person_and_calculate_distance

# ----------------------
# Global Variables
# ----------------------
original_img = None  # Will store the original loaded image
hog_img = None  # This should be assigned your HoG image after processing
roi_img = None  # This should be assigned your ROI image after processing
displayed_original = None  # Used for the label
displayed_processed = None  # Used for the label (either HoG or ROI)
original_image_path = None
current_ROI_file_name = None
is_left_hand = None
current_ROI_file_path = None
correct_identification = None


def choose_file():
    global original_img, lbl_left_img, original_image_path, is_left_hand

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
    )

    original_image_path = file_path

    if "_l_" in original_image_path:
        is_left_hand = True
    else:
        is_left_hand = False

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
    global roi_img, lbl_right_img, current_ROI_file_name, current_ROI_file_path

    if not original_image_path:
        return

    proie = PROIE()

    proie.extract_roi(original_image_path, rotate=True)
    roi_img = proie.roi_img

    # save ROI img
    palm_file_name = original_image_path.split("/")[-1].replace(".jpg", "")
    current_ROI_file_name = f"proie_roi_img_{palm_file_name}.jpg"
    proie.save(current_ROI_file_name)

    # save full path for HOG
    current_folder = os.path.dirname(os.path.abspath(__file__))
    current_ROI_file_path = os.path.join(current_folder, current_ROI_file_name)
    print("[DEBUG]: ", current_ROI_file_path)

    if len(roi_img.shape) == 2:
        pil_image = Image.fromarray(roi_img)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))

    fixed_size = (400, 400)
    pil_image = pil_image.resize(fixed_size)

    tk_roi = ImageTk.PhotoImage(pil_image)
    lbl_right_img.config(image=tk_roi)
    lbl_right_img.image = tk_roi


# def show_hog():
#     global hog_img, lbl_right_img, current_ROI_file_name, original_image, is_left_hand, current_ROI_file_path

#     print(original_img)
#     print(current_ROI_file_name)
#     if original_img is None or current_ROI_file_name is None:
#         messagebox.showerror(
#             "m ngu", "You need to choose a picture and perform ROI on it first."
#         )
#         return

#     orientation = "l" if is_left_hand else "r"

#     print(f"[DEBUG]: Current roi path: {current_ROI_file_path}")
#     print(f"[DEBUG]: Current orientation: {orientation}")
#     hog_img = process_hog(current_ROI_file_path, orientation, visualize=True)

#     # Convert the HOG image (grayscale) to a format Tkinter can display
#     pil_image = Image.fromarray(hog_img)

#     # Convert to ImageTk format
#     tk_hog = ImageTk.PhotoImage(pil_image)

#     # Update the label widget with the new image
#     lbl_right_img.config(image=tk_hog)
#     lbl_right_img.image = tk_hog  # Keep a reference to prevent garbage collection


def show_hog():
    global hog_img, lbl_right_img, current_ROI_file_name, original_image, is_left_hand, current_ROI_file_path

    if original_img is None or current_ROI_file_name is None:
        messagebox.showerror(
            "Error", "You need to choose a picture and perform ROI on it first."
        )
        return

    orientation = "l" if is_left_hand else "r"

    print(f"[DEBUG]: Current ROI path: {current_ROI_file_path}")
    print(f"[DEBUG]: Current orientation: {orientation}")

    hog_img = process_hog(current_ROI_file_path, orientation, visualize=False)

    def apply_colormap(image, colormap=cm.viridis):
        """
        Apply a colormap to a grayscale image and convert it to RGB.
        """
        colored_image = colormap(image / 255.0)
        return (colored_image[:, :, :3] * 255).astype("uint8")  # Convert to 8-bit RGB

    hog_img_colored = apply_colormap(hog_img)

    # Convert the colored HOG image to a PIL Image
    pil_image = Image.fromarray(hog_img_colored)

    # Resize the image for better display in Tkinter (optional scaling)
    scale_factor = 2  # Adjust the scaling factor as needed
    width, height = pil_image.size
    pil_image = pil_image.resize(
        (width * scale_factor, height * scale_factor), Image.Resampling.LANCZOS
    )

    tk_hog = ImageTk.PhotoImage(pil_image)
    lbl_right_img.config(image=tk_hog)
    lbl_right_img.image = tk_hog  # Keep a reference to prevent garbage collection


def predict():
    global current_ROI_file_path, correct_identification
    threshold = 9
    predicted_person, some_value_calc = find_person_and_calculate_distance(current_ROI_file_path)
    correct_identification = predicted_person
    print(predicted_person)
    print(some_value_calc)

    if some_value_calc > threshold:
        messagebox.showerror("Biometrics Demonstration App", f"Your palm is not in our database. Unable to authorize.\nGabor distance matching value: {some_value_calc}")
        print(f"Unable to authorize with distance value {some_value_calc}")
    else:
        messagebox.showinfo("Biometrics Demonstration App", f"Please enter your identification to proceed.\nGabor distance matching value: {some_value_calc}")
        print(f"With distance value {some_value_calc}, please enter your identification.")
        entry_username.configure(state="normal")
    return


def login():
    global correct_identification
    if entry_username.cget("state") == "disabled":
        messagebox.showerror("Error", "Please perform prediction first.")
        return

    password = entry_username.get()

    if password == correct_identification:
        messagebox.showinfo("Biometrics Demonstration App", "Logged in successfully!")
    else:
        messagebox.showerror("Biometrics Demonstration App", "Incorrect password. Please try again.")


def reset():
    print("reset done")


def main():
    global original_image, processed_image, lbl_left_img, lbl_right_img, entry_username, placeholder_img, placeholder_tk_img

    root = tk.Toplevel()
    root.title("Biometrics Demonstration App")

    # Create a 400x400 single-color placeholder image
    placeholder_img = Image.new("RGB", (400, 400), color="gray")
    placeholder_tk_img = ImageTk.PhotoImage(placeholder_img)

    # ----------------------------
    # Top Frame: two sub-frames (left & right) with an arrow in between
    # ----------------------------
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    # Left Frame
    original_image = tk.Frame(top_frame)
    original_image.pack(side=tk.LEFT, padx=10)

    lbl_left_img = tk.Label(original_image)
    lbl_left_img.pack()
    lbl_left_img.config(image=placeholder_tk_img)  # Set placeholder image
    lbl_left_img.image = placeholder_tk_img  # Ensure reference is persistent

    # Arrow in the middle
    arrow_label = tk.Label(top_frame, text="→", font=("Arial", 24))
    arrow_label.pack(side=tk.LEFT, padx=10)

    # Right Frame
    processed_image = tk.Frame(top_frame)
    processed_image.pack(side=tk.LEFT, padx=10)

    lbl_right_img = tk.Label(processed_image)
    lbl_right_img.pack()
    lbl_right_img.config(image=placeholder_tk_img)  # Set placeholder image
    lbl_right_img.image = placeholder_tk_img  # Ensure reference is persistent

    # ----------------------------
    # Frame for "Choose Image" and "Reset" buttons
    # ----------------------------
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    # Button to choose image
    btn_choose = tk.Button(button_frame, text="Choose Image", command=choose_file)
    btn_choose.pack(side=tk.LEFT, padx=5)

    # Button to reset
    btn_reset = tk.Button(button_frame, text="Reset", command=reset)
    btn_reset.pack(side=tk.LEFT, padx=5)

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

    entry_username = tk.Entry(login_frame, width=20, state="disabled")
    entry_username.pack(side=tk.LEFT, padx=5)

    btn_login = tk.Button(login_frame, text="Login", command=login)
    btn_login.pack(side=tk.LEFT, padx=5)

    root.mainloop()


if __name__ == "__main__":
    main()
