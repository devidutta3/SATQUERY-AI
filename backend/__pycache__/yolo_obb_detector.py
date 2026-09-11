from ultralytics import YOLO
from pathlib import Path


class YOLOOBBDetector:

    def __init__(self, model_path="yolo11n-obb.pt"):
        print("Loading YOLO OBB model...")

        self.model = YOLO(model_path)

        print("YOLO OBB model loaded successfully!")

    def detect(
        self,
        image_path,
        conf_threshold=0.25,
        output_path=None
    ):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False
        )

        result = results[0]

        if output_path is None:
            output_path = (
                image_path.parent /
                "yolo_obb_result.jpg"
            )

        output_path = Path(output_path)

        result.save(
            filename=str(output_path)
        )

        detections = []

        if result.obb is not None:

            for i in range(len(result.obb)):

                class_id = int(
                    result.obb.cls[i]
                )

                confidence = float(
                    result.obb.conf[i]
                )

                class_name = result.names[
                    class_id
                ]

                polygon = (
                    result.obb.xyxyxyxy[i]
                    .cpu()
                    .tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "polygon": polygon
                })

        return {
            "model": "YOLO-OBB",
            "image": str(image_path),
            "output_image": str(output_path),
            "num_objects": len(detections),
            "detections": detections
        }


if __name__ == "__main__":

    detector = YOLOOBBDetector()

    result = detector.detect(
        "test.jpg",
        conf_threshold=0.25
    )

    print("\nDetection Result:")
    print(result)