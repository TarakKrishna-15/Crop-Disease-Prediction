import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = r"D:\CSP\saved_models\best_model.h5"

# Map classes according to your model output order
CLASS_NAMES = ["Healthy", "Powdery Mildew", "Rust", "Leaf Spot"]

SUGGESTIONS = {
    "Healthy": [
        "✅ Keep regular watering.",
        "☀️ Provide adequate sunlight.",
        "🌱 Continue using balanced fertilizer."
    ],
    "Powdery Mildew": [
        "⚠️ Avoid overhead watering.",
        "🍃 Improve air circulation around plants.",
        "🧴 Use a sulfur-based fungicide spray."
    ],
    "Rust": [
        "🌿 Remove infected leaves immediately.",
        "☀️ Ensure good sunlight exposure.",
        "🧴 Apply neem oil or a mild fungicide."
    ],
    "Leaf Spot": [
        "💧 Water at the base, not on leaves.",
        "🪴 Use copper-based fungicides.",
        "🌞 Ensure proper drying after watering."
    ]
}

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def generate_gradcam(model, img_array, layer_name):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            output = predictions[:, class_idx]

        grads = tape.gradient(output, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM failed: {e}")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")
st.title("🌿 Plant Disease Detection using CNN + Grad-CAM")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("❌ Invalid image. Please upload a valid file.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Leaf", use_column_width=True)

        img_resized = cv2.resize(img_rgb, (128, 128))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        model = load_model()
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        predicted_class = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class {class_idx}"

        # 🟢 Result section
        st.subheader(f"🧠 Prediction: {predicted_class} ({confidence:.2f}%)")

        if predicted_class.lower() == "healthy":
            st.success("🌱 This plant is **Healthy!** Great job maintaining it! 💚")
        else:
            st.error(f"🚨 This plant is affected by **{predicted_class}**.")
            st.markdown("### 🩵 Tips to Make It Healthy Again:")
            for tip in SUGGESTIONS.get(predicted_class, ["No specific suggestions available."]):
                st.write(f"- {tip}")

        # 🔥 Grad-CAM
        heatmap = generate_gradcam(model, img_array, layer_name="conv2d")
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_rgb, 0.7, heatmap, 0.3, 0)
            st.subheader("🔥 Grad-CAM Visualization")
            st.image(superimposed_img, use_column_width=True)
        else:
            st.warning("⚠️ Grad-CAM could not be generated.")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("📤 Please upload a leaf image to start prediction.")
