from ultralytics import YOLO
from pathlib import Path


class YOLODetector:
    """
    Reusable YOLO-OBB detector for single-image inference.
    """

    def __init__(
        self,
        model_path="runs/obb/train/weights/best.pt"
    ):
        print("Loading YOLO-OBB model...")

        self.model = YOLO(model_path)

        print("YOLO-OBB model loaded successfully!")
        print(f"Task: {self.model.task}")
        print(f"Classes: {self.model.names}")

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
            output_path = image_path.parent / "yolo_result.jpg"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        result.save(
            filename=str(output_path)
        )

        print(
            f"Detection image saved to: {output_path}"
        )

        detections = []

        if result.obb is not None:

            obb = result.obb

            for i in range(len(obb)):

                points = obb.xyxyxyxy[i].tolist()
                xywhr = obb.xywhr[i].tolist()

                confidence = float(
                    obb.conf[i]
                )

                class_id = int(
                    obb.cls[i]
                )

                class_name = result.names[
                    class_id
                ]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "obb_points": [
                        [
                            round(point[0], 2),
                            round(point[1], 2)
                        ]
                        for point in points
                    ],
                    "xywhr": [
                        round(value, 2)
                        for value in xywhr
                    ]
                })

        return {
            "model": "YOLO-OBB",
            "task": "oriented_object_detection",
            "image": str(image_path),
            "output_image": str(output_path),
            "num_objects": len(detections),
            "detections": detections
        }


if __name__ == "__main__":

    detector = YOLODetector()

    result = detector.detect(
        image_path="test.jpg",
        conf_threshold=0.10,
        output_path="yolo_result.jpg"
    )

    print("\nDetection Result:")
    print("=" * 60)
    print(result)