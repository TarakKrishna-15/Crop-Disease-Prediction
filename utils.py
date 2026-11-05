import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

def preprocess_image(image, target_size=(128,128)):
    """Preprocess PIL Image or numpy array"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize(target_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image).astype('float32')/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_gradcam(model, img_array, last_conv_layer_name='last_conv'):
    """Generate Grad-CAM heatmap"""
    preds = model(img_array)
    pred_index = np.argmax(preds[0])

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs.numpy()[0]
    heatmap = conv_outputs @ pooled_grads.numpy()
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap
