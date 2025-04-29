from ultralytics import YOLO

# Pastikan kode utama berada di dalam blok 'if __name__ == "__main__":'
if __name__ == '__main__':
    # Load a pretrained YOLOv11n model
    model = YOLO("yolov11s.pt")  # Bisa juga ganti ke "yolov11s.pt" kalau mau sedikit lebih berat

    # Train the model on your custom dataset
    results = model.train(
        data="data.yaml",  # ganti dengan path data.yaml kamu
        epochs=100,
        imgsz=640,
        batch=16,                # 8–16 aman buat RTX 3050 8GB
        device=0,                # 0 = GPU (RTX 3050 kamu)
        workers=4,               # lebih cepat untuk loading data
        optimizer="SGD",         # optimasi lebih ringan dan stabil
        save=True,               # otomatis save model best.pt
        project="runs/train",    # folder output
        name="exp_uang"          # nama eksperimen
    )

    # Setelah training, kamu bisa evaluasi model
    metrics = model.val()

    # Coba infer ke 1 gambar
    results = model("satu_gambar.jpg")
    results[0].show()

    # Export model ke ONNX untuk mobile/deployment
    model.export(format="onnx")
