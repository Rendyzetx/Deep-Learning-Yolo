from ultralytics import YOLO

# Pastikan kode utama berada di dalam blok 'if __name__ == "__main__":'
if __name__ == '__main__':
    # Load model yang sudah dilatih (gunakan path ke model yang sudah dilatih, misalnya best.pt)
    model = YOLO("runs/train/exp_uang/weights/best.pt")  # Ganti dengan path ke model yang sudah dilatih

    # Export model ke format ONNX
    model.export(format="onnx")  # Format bisa diubah sesuai kebutuhan (misalnya 'onnx', 'tflite', dll.)

    print("Model diekspor ke format ONNX.")
