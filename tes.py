from ultralytics import YOLO


if __name__ == '__main__':
    # Load the trained model (gunakan model yang sudah dilatih, misalnya best.pt)
    model = YOLO("runs/train/exp_uang/weights/best.pt")  # Ganti dengan path ke model yang sudah dilatih

    # Coba infer ke 1 gambar
    results = model("tes3.jpg")  # Ganti dengan path ke gambar yang ingin diuji

    # Tampilkan hasil deteksi
    results[0].show()  

    # menyimpan hasil gambar deteksi
    results[0].save()  
