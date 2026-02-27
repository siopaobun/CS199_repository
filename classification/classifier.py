import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pandas as pd

model_path = 'models/Phytoplankton_EfficientNetV2B0/ckpts/final_model.h5'
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def preprocess_ifcb_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
        
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    
    img = cv2.resize(img, (100, 100))
    return img

ifcb_data_dir = 'inverted_images/'  
image_paths = glob.glob(os.path.join(ifcb_data_dir, '*.png'))

print(f"Found {len(image_paths)} images. Processing...")

images_batch = []
valid_paths = []

for path in image_paths:
    processed_img = preprocess_ifcb_image(path)
    if processed_img is not None:
        images_batch.append(processed_img)
        valid_paths.append(path)

X_batch = np.array(images_batch)

predictions = model.predict(X_batch)

predicted_class_indices = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)

filenames = [os.path.basename(p) for p in valid_paths]

results_df = pd.DataFrame({
    'Filename': filenames,
    'Predicted_Class_Index': predicted_class_indices,
    'Confidence': confidence_scores
})

output_csv = 'classification_results.csv'
results_df.to_csv(output_csv, index=False)
print(f"\nSaved all results successfully to {output_csv}")