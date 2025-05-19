from ultralytics import YOLO


if __name__ == '__main__':
    # Load a pretrained YOLOv11n model
    model = YOLO("yolov11s.pt") 

    # Train the model on your custom dataset
    results = model.train(
        data="data.yaml",  
        epochs=100,
        imgsz=640,
        batch=16,                
        device=0,                
        workers=4,               
        optimizer="SGD",         
        save=True,               
        project="runs/train",    
        name="exp_uang"          
    )

    # evaluasi model
    metrics = model.val()
