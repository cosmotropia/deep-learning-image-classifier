import html

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------
# Clases
# -------------------------
classes = [
    "T-shirt", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]


def load_images(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_labels(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data


def show_centered_result(image_input, caption_text, pred_class, confidence):
    st.markdown('<div style="height: 2.5rem;"></div>', unsafe_allow_html=True)
    _, c, _ = st.columns([1, 2, 1])
    with c:
        i1, i2, i3 = st.columns([1, 4, 1], vertical_alignment="center")
        with i2:
            st.image(image_input, width=150)
        cap = html.escape(caption_text)
        pred_label = html.escape(classes[pred_class])
        st.markdown(
            f"""
            <div style="text-align: center; padding-bottom: 1.5rem;">
                <p style="margin-top: 0.75rem; font-size: 0.95rem;">{cap}</p>
                <h3 style="margin-top: 1rem;">Predicción: {pred_label}</h3>
                <p style="font-size: 1.1rem;">Confianza: {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_resource
def load_fashion_model():
    return load_model("fashion_model.h5")


@st.cache_data
def load_test_dataset():
    X_test = load_images("data/t10k-images-idx3-ubyte")
    y_test = load_labels("data/t10k-labels-idx1-ubyte")
    return X_test, y_test


st.title("👕 Clasificador de ropa")

with st.spinner("Cargando modelo…"):
    model = load_fashion_model()

# -------------------------
# Selección de modo
# -------------------------
option = st.radio(
    "Selecciona una opción:",
    ["Subir imagen", "Usar imagen de prueba"],
)

# -------------------------
# OPCIÓN 1: Upload
# -------------------------
if option == "Subir imagen":

    uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((28, 28))

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        with st.spinner("Clasificando imagen…"):
            prediction = model.predict(img_array, verbose=0)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        show_centered_result(image, "Imagen subida", pred_class, confidence)

# -------------------------
# OPCIÓN 2: Dataset LOCAL
# -------------------------
else:

    with st.spinner("Cargando imágenes de prueba…"):
        X_test, y_test = load_test_dataset()

    max_i = len(X_test) - 1
    if "img_slider" not in st.session_state:
        st.session_state.img_slider = 0
    st.session_state.img_slider = int(
        min(max_i, max(0, st.session_state.img_slider))
    )

    st.markdown("**Selecciona una imagen**")
    col_prev, col_slider, col_next = st.columns(
        [1, 8, 1],
        vertical_alignment="center",
    )
    with col_prev:
        if st.button("◀", key="nav_prev", help="Imagen anterior"):
            st.session_state.img_slider = max(0, st.session_state.img_slider - 1)
    with col_next:
        if st.button("▶", key="nav_next", help="Imagen siguiente"):
            st.session_state.img_slider = min(max_i, st.session_state.img_slider + 1)

    with col_slider:
        index = st.slider(
            "Selecciona una imagen",
            0,
            max_i,
            key="img_slider",
            label_visibility="collapsed",
        )

    image = X_test[index]
    label = y_test[index]

    img_array = image / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    with st.spinner("Clasificando imagen…"):
        prediction = model.predict(img_array, verbose=0)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    show_centered_result(
        image,
        f"Real: {classes[label]}",
        pred_class,
        confidence,
    )
