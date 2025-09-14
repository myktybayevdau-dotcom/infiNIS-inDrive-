from ultralytics import YOLO

def main():
    # Загружаем модель детекции (маленькая, быстрая)
    model = YOLO("yolo11n.pt")

    # Явно указываем задачу как detection
    model.train(
        data=r"C:\Users\Dauren\Downloads\Car Scratch and Dent.v5i.yolov11\data.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        task="detect"   # 🔑 это главное!
    )

if __name__ == "__main__":
    main()
