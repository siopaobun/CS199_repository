import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pandas as pd

model_path = 'models/Phytoplankton_EfficientNetV2B0/ckpts/final_model.h5'
model = tf.keras.models.load_model(model_path)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ordered list of class names (index -> class label)
class_names = [
    "Appendicularia",
    "Actinoptychus_senarius",
    "Actinoptychus",
    "Bacillariophyceae_type1_colony",
    "Bacillaria_paxillifer",
    "Bacteriastrum",
    "Bacillariophyceae",
    "Aulacodiscus_argus",
    "Actinoptychus_splendens",
    "Bellerochea_horologicalis",
    "Bellerochea",
    "Asterionella",
    "Artefact",
    "Biddulphia_alternans",
    "Biddulphianae",
    "Brockmanniella_brockmannii",
    "Bubbles",
    "Centric_Diatom",
    "Cerataulus_granulata",
    "Ceratium_horridum+C._longipes",
    "Chaetoceros",
    "Chaetoceros_affinis",
    "Chaetoceros_curvisetus+C._pseudocurvisetus",
    "Chaetoceros_danicus",
    "Chaetoceros_socialis",
    "Ciliophora",
    "Cnidaria",
    "Copepoda_adult",
    "Coscinodiscus_concinnus",
    "Coscinodiscus_granii",
    "Crustacea",
    "Crustaceae-part",
    "Dactyliosolen+Cerataulina+Guinardia",
    "Detritus",
    "Dinoflagellata",
    "Dinoflagellate_cyst",
    "Ditylum_brightwellii",
    "Diploneis",
    "Egg+Cyst",
    "Eucampia",
    "Faecal_pellet",
    "Favella",
    "Foraminifera",
    "Leptocylindraceae",
    "Lithodesmium_undulatum",
    "Hobaniella_longicruris",
    "Melosira",
    "Helicotheca_tamesis",
    "Guinardia_delicatula",
    "Guinardia_striata+Dactyliosolen_phuketensis",
    "Fibers",
    "Guinardia_flaccida",
    "Lauderia+Melosira+Detonula",
    "Meuniera_membranacea",
    "Mollusca",
    "Nauplii",
    "Neocalyptrella_robusta",
    "Noctiluca_scintillans",
    "Noctilucales",
    "Odontella_aurita+Ralfsiella_minima",
    "Odontella_rhombus_f._trigona",
    "Paralia",
    "Pennate_Diatom",
    "Pennate_Diatom_colony",
    "Peritrichia",
    "Phytoplankton_Colony",
    "Plagiogrammopsis+Bellerochea_malleus",
    "Polychaeta",
    "Pollen",
    "Porifera_spicule",
    "Proboscia_alata",
    "Proboscia_indica",
    "Protoperidinium",
    "Protoperidinium_pentagonum",
    "Pseudo-nitzschia",
    "Remnant",
    "Rhizosolenia",
    "Rhizosolenia_setigera_(f._pungens)+R._hebetata_f._semispina",
    "Rotifera",
    "Skeletonema",
    "Stellarima_stellaris+Podosira+Hyalodiscus",
    "Stephanopyxis",
    "Suctoria",
    "Synedra+Thalassionema",
    "Thalassiosira+Porosira",
    "Tintinnina",
    "Tintinnopsis",
    "Triceratium_favus",
    "Trieres_mobiliensis+T._regia",
    "Trieres_sinensis",
    "Tripos",
    "Tripos_fusus",
    "Veliger_larvae_D-shaped",
    "Zooplankton",
    "Zygoceros"
]

def preprocess_ifcb_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
        
    # if img.dtype == np.uint16:
    #     img = (img / 256).astype(np.uint8)
    
    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    img /= 255.0
    
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

predicted_labels = [class_names[i] for i in predicted_class_indices]

results_df = pd.DataFrame({
    'Filename': filenames,
    'Predicted_Class_Index': predicted_class_indices,
    'Predicted_Class': predicted_labels,
    'Confidence': confidence_scores
})

output_csv = 'classification_results.csv'
results_df.to_csv(output_csv, index=False)
print(f"\nSaved all results successfully to {output_csv}")