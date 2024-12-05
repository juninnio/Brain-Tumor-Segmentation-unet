import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import os
import nibabel as nib
from keras.saving import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import zoom

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def get_model():
    @tf.keras.utils.register_keras_serializable()
    def dice_coef(y_true, y_pred):
        y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

    @tf.keras.utils.register_keras_serializable()
    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    @tf.keras.utils.register_keras_serializable()
    def iou(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        union = total - intersection
        return (intersection + smooth) / (union + smooth)

    @tf.keras.utils.register_keras_serializable()
    def per_class_dice(class_index):
        def dice(y_true, y_pred):
            # Convert the true labels to a one-hot encoding (if needed)
            y_true_class = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), class_index), tf.float32)
            
            # Convert the predicted labels to a one-hot encoding
            y_pred_class = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), class_index), tf.float32)
            
            # Flatten the arrays
            y_true_f = tf.keras.backend.flatten(y_true_class)
            y_pred_f = tf.keras.backend.flatten(y_pred_class)
            
            # Calculate intersection and Dice coefficient
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6)
    
        return dice

    model = load_model('./3d brain tumor segmentation_attention_unet.keras')
    return model

def load_preprocess_image(folder_dir):
    modalities = os.listdir(folder_dir)
    for modality in modalities:
        if 'flair' in modality:
            path = os.path.join(folder_dir, modality)
            flair_img = np.array(nib.load(path).get_fdata()).astype(np.float32)

        elif 't2' in modality:
            path = os.path.join(folder_dir, modality)
            t2_img = np.array(nib.load(path).get_fdata()).astype(np.float32)

        elif 't1ce' in modality:
            path = os.path.join(folder_dir, modality)
            t1ce_img = np.array(nib.load(path).get_fdata()).astype(np.float32)

        elif 'seg' in modality:
            path = os.path.join(folder_dir, modality)
            seg_img = np.array(nib.load(path).get_fdata())

    flair = scaler.fit_transform(flair_img.reshape(-1, flair_img.shape[-1])).reshape(flair_img.shape)
    flair = flair[56:184, 56:184, 14:142]

    t1ce = scaler.fit_transform(t1ce_img.reshape(-1, t1ce_img.shape[-1])).reshape(t1ce_img.shape)
    t1ce = t1ce[56:184, 56:184, 14:142]

    t2 = scaler.fit_transform(t2_img.reshape(-1, t2_img.shape[-1])).reshape(t2_img.shape)
    t2 = t2[56:184, 56:184, 14:142]

    if 4 in seg_img:
        seg_img[seg_img==4] = 3
    
    seg = seg_img.astype(np.uint8)
    seg = seg[56:184, 56:184,14:142]
    seg = to_categorical(seg, num_classes=4)

    combined = np.stack([t1ce, t2, flair], axis=-1)
    
    img_input = np.expand_dims(combined, axis=0)
    test_mask = np.argmax(seg, axis=-1)
    return img_input, test_mask, t1ce

def get_prediction(model, image):
    prediction = model.predict(image)
    pred_binary = np.argmax(prediction, axis=-1)
    pred_binary = pred_binary.squeeze(0)

    final_mask = np.zeros_like(pred_binary)
    final_mask[pred_binary == 1] = 1
    final_mask[pred_binary == 2] = 2
    final_mask[pred_binary == 3] = 3

    return final_mask

def update_image(image_data, slice_number, img):
    ax.clear()
    cmap = colors.ListedColormap(['black','red','pink','blue'])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(img[:,:,slice_number],cmap='gray')
    ax.imshow(image_data[:,:,slice_number], cmap=cmap, norm=norm, alpha=0.5)
    legend_patches = [
        mpatches.Patch(color='red', label='Tumor Core'),
        mpatches.Patch(color='pink', label='Whole Tumor'),
        mpatches.Patch(color='blue', label='Enhancing Tumor')
    ]
    plt.legend(handles=legend_patches, loc="lower right")
    canvas.draw()

def on_predict():
    global predicted_image, original
    img, mask, original = load_preprocess_image(folder_path.get())
    model = get_model()
    predicted_image = get_prediction(model, img)

    update_image(predicted_image, 64, original)

    slice_slider.config(state=tk.NORMAL)
    slice_slider.set(64)

    if not canvas_widget.winfo_ismapped():
        canvas_widget.pack()

def select_folder():
    selected_folder = filedialog.askdirectory()
    folder_path.set(selected_folder)

def on_slice_change(val):
    slice_number = int(val)
    update_image(predicted_image, slice_number, original)


root = tk.Tk()
root.title('Brain Tumor Segmentation')
root.geometry("1000x1000")


folder_path = tk.StringVar()
tk.Label(root, text="Select Folder:").pack(pady=10)
folder_button = tk.Button(root, text="Browse", command=select_folder)
folder_button.pack(pady=5)

# Predict button
predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack(pady=10)

slice_slider = tk.Scale(root, from_=0, to=127, orient=tk.HORIZONTAL, label="Select Slice", command=on_slice_change)
slice_slider.set(64)
slice_slider.pack(pady=10)
slice_slider.config(state=tk.DISABLED)


# Frame for the image and controls
image_frame = tk.Frame(root)
image_frame.pack(padx=10, pady=10)

# Matplotlib figure and canvas to display the image slice
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title("Slice 64")
canvas = FigureCanvasTkAgg(fig, master=image_frame)
canvas_widget = canvas.get_tk_widget()


root.mainloop()

