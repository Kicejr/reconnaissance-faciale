from deepface import DeepFace
from deepface.modules.verification import find_distance
import cv2
import pickle


# Initialisation de la caméra avec une résolution plus basse pour accélérer la capture
cap = cv2.VideoCapture(0)  # Utilisation de la webcam par défaut
cap.set(3, 640)  # Largeur
cap.set(4, 480)  # Hauteur

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Paramètres pour la détection faciale
model_name = "Facenet512"
metrics = [{"cosine": 0.30}, {"euclidean": 20.0}, {"euclidean_l2": 0.78}]

# Chargement des embeddings existants si disponibles
try:
    with open("./embeddings/embs_facenet512.pkl", "rb") as file:
        embs = pickle.load(file)
        print("Existing embeddings file loaded successfully.")
except FileNotFoundError:
    print("No existing embeddings file found. Check your path.")
    exit()

# Normalisation avec CLAHE
def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Détection faciale avec DeepFace
def detect_faces_deepface(frame):
    try:
        # Détection avec DeepFace
        results = DeepFace.extract_faces(
            frame, detector_backend="mtcnn", enforce_detection=False
        )
        detected_faces = []
        for result in results:
            if result["confidence"] >= 0.5:
                x, y, w, h = result["facial_area"]["x"], result["facial_area"]["y"], result["facial_area"]["w"], result["facial_area"]["h"]
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Redimensionnement et traitement
                cropped_face = frame[y: y + h, x: x + w]
                cropped_face_resized = cv2.resize(cropped_face, (224, 224))
                cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
                cropped_face_norm = clahe(cropped_face_gray)
                cropped_face_gray = cv2.cvtColor(cropped_face_norm, cv2.COLOR_GRAY2RGB)

                emb = DeepFace.represent(
                    cropped_face_gray,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]

                # Comparaison avec les embeddings existants
                min_dist = float("inf")
                match_name = None
                for name, emb2 in embs.items():
                    dst = find_distance(emb, emb2, list(metrics[2].keys())[0])
                    if dst < min_dist:
                        min_dist = dst
                        match_name = name

                # Si correspondance trouvée
                if min_dist < list(metrics[2].values())[0]:
                    detected_faces.append(
                        (x1, y1, x2, y2, match_name, min_dist, (0, 255, 0))
                    )
                    print(f"Detected as: {match_name} {min_dist:.2f}")
                else:
                    detected_faces.append(
                        (x1, y1, x2, y2, "Inconnu", min_dist, (0, 0, 255))
                    )
        return detected_faces
    except Exception as e:
        print(f"Error during face detection: {e}")
        return []


# Boucle principale avec détection limitée à chaque 10 images
frame_count = 0
detected_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not read successfully")
        break

    # Effectuer la détection toutes les 10 itérations
    if frame_count % 10 == 0:
        detected_faces = detect_faces_deepface(frame)

    # Afficher les visages détectés
    for x1, y1, x2, y2, name, min_dist, color in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {min_dist:.2f}",
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Afficher la fenêtre principale
    cv2.imshow("Optimized Camera Feed", frame)

    # Attente pour quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1  # Compteur d'images pour limiter la fréquence de détection

cap.release()
cv2.destroyAllWindows()
