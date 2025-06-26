import cv2
from ultralytics import YOLO
from tqdm import tqdm


def detect_people_in_video(input_path: str, output_path: str) -> None:
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.4, verbose=False)

        # результируем один кадр
        result = results[0]

        # выполняем фильрацию по class Person
        names = model.model.names
        person_class_id = [k for k, v in names.items() if v == "person"][0]

        # исключение лишних объектов
        if result.boxes is not None:
            result.boxes = result.boxes[result.boxes.cls == person_class_id]

        # Отрисовка
        annotated_frame = result.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"[INFO] Готово! Видео сохранено как {output_path}")
