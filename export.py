from ultralytics import YOLO


if __name__ == '__main__':
    # Load model 
    model = YOLO("runs/train/exp_uang/weights/best.pt")  

    # Export model ke format ONNX
    model.export(format="onnx")  

    print("Model diekspor ke format ONNX.")
