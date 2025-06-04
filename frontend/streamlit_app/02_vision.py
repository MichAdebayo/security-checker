"""
Streamlit application for real‑time PPE detection using the Roboflow
Inference API and your webcam.

Prerequisites
-------------
➜  pip install streamlit streamlit-webrtc opencv-python-headless inference-sdk numpy

Then run:
➜  streamlit run streamlit_ppe_detection.py
"""

import time
from typing import List

import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# -----------------------------------------------------------------------------
# 1. CONFIGURATION PAR DÉFAUT
# -----------------------------------------------------------------------------
API_URL_DEFAULT = "https://detect.roboflow.com"
API_KEY_DEFAULT = "mDauQAfDrFWieIsSqti6"  # ↩️ Remplacez par votre propre clé !
MODEL_ID_DEFAULT = "ppe-factory-bmdcj/2"    # ↩️ Format « <projet>/<version> »
CONF_THRESH_DEFAULT = 0.5
OVERLAP_THRESH_DEFAULT = 0.3
DETECTION_INTERVAL_DEFAULT = 5.0  # secondes


st.title("🦺 Détection d'EPI en direct avec Roboflow")

# ---- SIDEBAR --------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    api_key = st.text_input("Roboflow API key", value=API_KEY_DEFAULT, type="password")
    model_id = st.text_input("Model ID", value=MODEL_ID_DEFAULT)

    conf_thresh = st.slider("Seuil de confiance", 0.0, 1.0, CONF_THRESH_DEFAULT, 0.05)
    overlap_thresh = st.slider("Seuil IoU (NMS)", 0.0, 1.0, OVERLAP_THRESH_DEFAULT, 0.05)

    detection_interval = st.number_input(
        "Intervalle d'inférence (s)",
        min_value=0.1,
        max_value=30.0,
        value=DETECTION_INTERVAL_DEFAULT,
        step=0.1,
        format="%0.1f",
    )

    start_webcam = st.checkbox("🎥 Activer la webcam", value=False)
    st.markdown("""---
    **Astuce :** pour économiser votre quota API, choisissez un *intervalle d'inférence* plus grand.
    """)

# -----------------------------------------------------------------------------
# 3. UTILITÉ : NMS (Non‑Max Suppression)
# -----------------------------------------------------------------------------

def apply_nms(
    preds: List[dict], conf_thresh: float, overlap_thresh: float
) -> List[dict]:
    """Filtre d'abord par confiance, puis applique un NMS OpenCV."""

    boxes, confidences, raw_indices = [], [], []

    for idx, p in enumerate(preds):
        conf = p["confidence"]
        if conf >= conf_thresh:
            cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            boxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(conf))
            raw_indices.append(idx)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, overlap_thresh)

    kept = []
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        kept.append(preds[raw_indices[i]])

    return kept

# -----------------------------------------------------------------------------
# 4. CLIENT ROBOFLOW (mis en cache)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_rf_client(api_key: str):
    return InferenceHTTPClient(api_url=API_URL_DEFAULT, api_key=api_key)

# -----------------------------------------------------------------------------
# 5. VIDEO TRANSFORMER
# -----------------------------------------------------------------------------

class PPETransformer(VideoTransformerBase):
    """Traite chaque frame pour annoter les EPI détectés."""

    def __init__(
        self,
        api_client: InferenceHTTPClient,
        model_id: str,
        conf_thresh: float,
        overlap_thresh: float,
        detection_interval: float,
    ):
        self.client = api_client
        self.model_id = model_id
        self.conf_thresh = conf_thresh
        self.overlap_thresh = overlap_thresh
        self.detection_interval = detection_interval

        self._last_preds: List[dict] = []
        self._last_time = 0.0

    def _infer(self, image: np.ndarray):
        """Appelle l'API Roboflow selon l'intervalle défini."""
        current = time.time()
        if current - self._last_time >= self.detection_interval:
            try:
                response = self.client.infer(image, model_id=self.model_id)
                self._last_preds = response.get("predictions", [])
                self._last_time = current
            except Exception as e:
                # Streamlit-webrtc n'affiche pas directement d'exception ; on peut tracer sur le cadre.
                cv2.putText(
                    image,
                    f"API error: {e}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
        return self._last_preds

    def transform(self, frame):
        # streamlit-webrtc fournit un VideoFrame ↓↓↓
        img = frame.to_ndarray(format="bgr24")

        preds = self._infer(img)
        preds = apply_nms(preds, self.conf_thresh, self.overlap_thresh)

        # Dessine les seuils
        cv2.putText(
            img,
            f"Conf: {self.conf_thresh:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"IoU: {self.overlap_thresh:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Dessine chaque boîte retenue
        for p in preds:
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{p['class']} ({p['confidence']:.2f})"
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Retourne le frame au format RGB pour l'affichage
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------------------------------
# 6. LANCEMENT DU STREAM
# -----------------------------------------------------------------------------

if start_webcam:

    rf_client = get_rf_client(api_key)

    webrtc_streamer(
        key="ppe-detection-stream",
        video_processor_factory=lambda: PPETransformer(
            api_client=rf_client,
            model_id=model_id,
            conf_thresh=conf_thresh,
            overlap_thresh=overlap_thresh,
            detection_interval=detection_interval,
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    st.info("Activez l'option **Activer la webcam** dans la barre latérale pour commencer.")
