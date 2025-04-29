from ultralytics import YOLO

# Pastikan kode utama berada di dalam blok 'if __name__ == "__main__":'
if __name__ == '__main__':
    # Load the trained model (gunakan model yang sudah dilatih, misalnya best.pt)
    model = YOLO("runs/train/exp_uang/weights/best.pt")  # Ganti dengan path ke model yang sudah dilatih

    # Coba infer ke 1 gambar
    results = model("tes3.jpg")  # Ganti dengan path ke gambar yang ingin diuji

    # Tampilkan hasil deteksi
    results[0].show()  # Menampilkan gambar dengan bounding box

    # Jika kamu ingin menyimpan hasil gambar dengan deteksi
    results[0].save()  # Menyimpan gambar hasil inferensi
