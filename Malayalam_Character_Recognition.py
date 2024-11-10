import cv2
import numpy as np
import matplotlib.font_manager as fm
from tkinter import filedialog, Tk, Label, Button, Frame, messagebox
import tensorflow as tf
from PIL import Image, ImageTk

# Load custom Malayalam font for display
mal_font_path = 'NotoSansMalayalam-Regular.ttf'
mal_font = fm.FontProperties(fname=mal_font_path)

# Load the pre-trained CNN model
cnn_model = tf.keras.models.load_model('my_cnn_model.h5')

# Malayalam characters the model is trained on
mal_characters = ['അ', 'ആ', 'ഇ', 'ഉ', 'ഋ', 'എ', 'ഏ', 'ഒ', 'ക', 'ക്ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ങ്ക', 'ങ്ങ', 'ച', 'ച്ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ട്ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ണ്ട', 'ണ്ണ', 'ണ്\u200d', 'ത', 'ത്ത', 'ഥ', 'ദ', 'ധ', 'ന', 'പ', 'പ്പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'മ്പ', 'മ്മ', 'യ', 'യ്യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ൻ', 'ർ', 'ൽ', 'ൾ']

# Prediction threshold
CONF_THRESHOLD = 0.5

# Function to prepare image and make predictions
def predict_character(image_path):
    try:
        # Read and process the image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (64, 64))
        normalized_image = resized_image.astype('float32') / 255.0
        input_image = np.expand_dims(np.expand_dims(normalized_image, axis=-1), axis=0)

        # Model prediction
        preds = cnn_model.predict(input_image)
        class_index = np.argmax(preds, axis=1)[0]
        probability = np.max(preds)

        # Confirm prediction confidence
        if probability < CONF_THRESHOLD:
            return resized_image, None, probability
        else:
            predicted_char = mal_characters[class_index] if class_index < len(mal_characters) else None
            return resized_image, predicted_char, probability
    except Exception as error:
        print(f"Prediction error: {error}")
        return None, None, None

# Function to open an image file
def choose_file():
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
    )
    if file_path:
        processed_image, character, confidence = predict_character(file_path)
        if processed_image is not None:
            if character is None:
                result_display.config(text="Character unrecognized!", font=('Arial', 18, 'bold'), fg="red")
            else:
                result_display.config(
                    text=f"Character: {character}\nConfidence: {confidence:.2f}",
                    font=('Arial', 18, 'bold'), fg="#1e90ff"
                )
            show_image(processed_image)
        else:
            messagebox.showerror("Error", "Image processing failed.")


# Display the image in the application window
def show_image(processed_image):
    rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(rgb_image)

    # Resize the image to a larger size (e.g., 400x400 pixels)
    image_pil = image_pil.resize((400, 400), Image.Resampling.LANCZOS)

    image_tk = ImageTk.PhotoImage(image_pil)
    image_display.config(image=image_tk)
    image_display.image = image_tk


# Setup for the main application window
app = Tk()
app.title("Malayalam Character Recognition")
app.geometry("600x700")
app.config(bg="#39DFCF")

# Layout structure
main_frame = Frame(app, bg="#f4f4f9")
main_frame.pack(pady=40)

# Application title
title_display = Label(main_frame, text="Malayalam Character Prediction", font=('Arial', 24, 'bold'), bg="#f4f4f9", fg="#FF6347")
title_display.pack(pady=10)

# Button to select image
file_button = Button(main_frame, text="Select Image", command=choose_file, font=('Arial', 14, 'bold'), bg="#32CD32",
                     fg="white", width=20, relief="solid", bd=2)
file_button.pack(pady=15)

# Display result
result_display = Label(main_frame, text="Predicted Character:", font=('Arial', 18), bg="#f4f4f9")
result_display.pack(pady=20)

# Display area for selected image
image_display = Label(main_frame, bg="#f4f4f9")
image_display.pack(pady=20)


# Run application
app.mainloop()

# need = ['ങ്ങ', 'ൺ','ൽ','ങ്ങ','ഝ']
# mal_characters = ['അ', 'ആ', 'ഇ', 'ഉ', 'ഋ', 'എ', 'ഏ', 'ഒ', 'ക', 'ക്ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ങ്ക', 'ങ്ങ', 'ച', 'ച്ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ട്ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ണ്ട', 'ണ്ണ', 'ണ്\u200d', 'ത', 'ത്ത', 'ഥ', 'ദ', 'ധ', 'ന', 'പ', 'പ്പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'മ്പ', 'മ്മ', 'യ', 'യ്യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ൻ', 'ർ', 'ൽ', 'ൾ']
#
# for ch in need:
#     if ch in mal_characters:
#         print(f'{ch} = {mal_characters.index(ch)}')
