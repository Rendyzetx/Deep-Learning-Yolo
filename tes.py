from ultralytics import YOLO


if __name__ == '__main__':
    
    model = YOLO("runs/train/exp_uang/weights/best.pt")  

    # Coba infer ke 1 gambar
    results = model("tes3.jpg")  

    # Tampilkan hasil deteksi
    results[0].show()  

    # menyimpan hasil gambar deteksi
    results[0].save()  
