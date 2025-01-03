from deepface import DeepFace
import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop(input_dir, output_dir, detector_backend="mtcnn"):
    # Créer le dossier de sortie si il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        img_name = img_file.split(".")[0]

        try:
            # Extraction des visages avec color_face="rgb"
            faces = DeepFace.extract_faces(
                img_path,
                detector_backend=detector_backend,
                enforce_detection=True,
                color_face="rgb"  # Utilisation de "rgb" pour color_face
            )

            if faces:
                face = faces[0]["face"]  # Extraction du premier visage

                # Vérification des valeurs des pixels
                print(f"Valeurs des pixels avant traitement :\n{face[0:5, 0:5, :]}")

                # Si l'image est en float64, convertissons-la en uint8
                if face.dtype != 'uint8':
                    face = np.array(face * 255, dtype=np.uint8)  # Normalisation si nécessaire
                    print(f"Valeurs des pixels après conversion :\n{face[0:5, 0:5, :]}")

                # Redimensionnement de l'image
                face = cv2.resize(face, (224, 224))

                # Affichage avec Matplotlib
                plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title("Cropped Face")
                plt.show()

                # Sauvegarde de l'image
                cv2.imwrite(f"{output_dir}/{img_name}.jpg", face)
            else:
                print(f"Aucun visage détecté dans {img_file}")
        except Exception as e:
            print(f"Erreur avec {img_file}: {e}")


# Appel de la fonction avec le backend opencv
crop("./data", "./cropped_faces", detector_backend="mtcnn")
