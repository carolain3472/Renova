#!/usr/bin/env python3
"""
Water Hyacinth Species Classifier — Raspberry Pi
=================================================
Uso:
    python3 raspberry_pi_classifier.py                  # Modo cámara en vivo
    python3 raspberry_pi_classifier.py --image foto.jpg # Clasificar una imagen

Requisitos en Raspberry Pi:
    pip3 install tflite-runtime numpy Pillow
    pip3 install picamera2  # Para cámara (ya viene en Raspberry Pi OS)
"""

import argparse
import json
import time
import numpy as np
from PIL import Image

# Intentar importar tflite-runtime (más ligero) o tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def load_model(model_path, labels_path):
    """Carga modelo TFLite y metadatos."""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    with open(labels_path, 'r') as f:
        metadata = json.load(f)

    return interpreter, metadata


def preprocess_image(image_path, img_size, mean, std):
    """Preprocesa imagen para inferencia."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Normalizar con estadísticas de ImageNet
    img_array = (img_array - np.array(mean)) / np.array(std)

    # Agregar dimensión de batch: (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify(interpreter, img_array, metadata):
    """Ejecuta inferencia y retorna predicción."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ajustar tipo de dato si es INT8
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        img_array = img_array / input_scale + input_zero_point
        img_array = img_array.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img_array)

    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # ms

    output = interpreter.get_tensor(output_details[0]['index'])

    # Si es INT8, dequantizar
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - output_zero_point) * output_scale

    # Softmax
    exp_output = np.exp(output[0] - np.max(output[0]))
    probabilities = exp_output / exp_output.sum()

    pred_idx = np.argmax(probabilities)
    idx_to_class = {int(k): v for k, v in metadata['idx_to_class'].items()}

    return {
        'species': idx_to_class[pred_idx],
        'confidence': float(probabilities[pred_idx]) * 100,
        'inference_ms': inference_time,
        'all_probs': {idx_to_class[i]: float(p) * 100 for i, p in enumerate(probabilities)}
    }


def camera_loop(interpreter, metadata):
    """Modo cámara en vivo con Picamera2."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("Error: picamera2 no disponible. Instala con: sudo apt install python3-picamera2")
        return

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    print("Cámara iniciada. Presiona Ctrl+C para salir.\n")

    try:
        while True:
            # Capturar imagen
            frame = picam2.capture_array()
            img = Image.fromarray(frame).convert('RGB')

            # Preprocesar
            img_resized = img.resize((metadata['img_size'], metadata['img_size']))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = (img_array - np.array(metadata['mean'])) / np.array(metadata['std'])
            img_array = np.expand_dims(img_array, axis=0)

            # Clasificar
            result = classify(interpreter, img_array, metadata)

            print(f"Especie: {result['species']:30s} | "
                  f"Confianza: {result['confidence']:5.1f}% | "
                  f"Tiempo: {result['inference_ms']:6.1f} ms")

            time.sleep(2)  # Esperar 2 segundos entre capturas

    except KeyboardInterrupt:
        print("\nDetenido por usuario.")
    finally:
        picam2.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Water Hyacinth Classifier - Raspberry Pi')
    parser.add_argument('--model', default='water_hyacinth_int8.tflite', help='Ruta al modelo TFLite')
    parser.add_argument('--labels', default='labels.json', help='Ruta al archivo de labels')
    parser.add_argument('--image', default=None, help='Ruta a imagen (si no se da, usa cámara)')
    args = parser.parse_args()

    print("Cargando modelo...")
    interpreter, metadata = load_model(args.model, args.labels)
    print(f"Modelo cargado: {args.model}\n")

    if args.image:
        # Modo imagen individual
        img_array = preprocess_image(args.image, metadata['img_size'],
                                      metadata['mean'], metadata['std'])
        result = classify(interpreter, img_array, metadata)

        print(f"Imagen: {args.image}")
        print(f"Especie detectada: {result['species']}")
        print(f"Confianza: {result['confidence']:.1f}%")
        print(f"Tiempo de inferencia: {result['inference_ms']:.1f} ms")
        print(f"\nProbabilidades por clase:")
        for species, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
            bar = '█' * int(prob / 2)
            print(f"  {species:30s} {prob:5.1f}% {bar}")
    else:
        # Modo cámara en vivo
        camera_loop(interpreter, metadata)