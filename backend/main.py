from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from .yolo_detector import YOLODetector
from .burnscar_overlap_inference import run_burnscar_inference


app = FastAPI(title="SatQuery AI")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SERVE OUTPUT IMAGES
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

YOLO_MODEL = (
    BASE_DIR.parent
    / "runs"
    / "obb"
    / "train"
    / "weights"
    / "best.pt"
)

yolo_detector = YOLODetector(
    model_path=YOLO_MODEL
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SatQuery AI backend is running"
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    query: str = Form(...)
):

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    input_path = UPLOAD_DIR / image.filename

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(
            image.file,
            buffer
        )

    query_lower = query.lower()


    # ========================================================
    # PRITHVI — BURN SCAR DETECTION
    # ========================================================

    burn_keywords = [
        "burn",
        "burned",
        "burnt",
        "burn scar",
        "burn scars",
        "fire scar",
        "wildfire"
    ]

    if any(
        keyword in query_lower
        for keyword in burn_keywords
    ):

        # ----------------------------------------------------
        # Prithvi requires multispectral HLS GeoTIFF
        # ----------------------------------------------------

        if input_path.suffix.lower() not in [
            ".tif",
            ".tiff"
        ]:

            return {
                "task": "Burn Scar Segmentation",
                "query": query,
                "filename": image.filename,
                "error": (
                    "Prithvi BurnScars requires a "
                    "6-band HLS GeoTIFF "
                    "(BLUE, GREEN, RED, NIR_NARROW, "
                    "SWIR_1, SWIR_2). "
                    "The uploaded image is not a GeoTIFF."
                )
            }

        result = run_burnscar_inference(
            image_path=input_path,
            output_dir=OUTPUT_DIR
        )

        return {
            "task": "Burn Scar Segmentation",
            "query": query,
            "filename": image.filename,
            "result": result
        }


    # ========================================================
    # YOLO — OBJECT DETECTION
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"yolo_{image.filename}"
    )

    result = yolo_detector.detect(
        image_path=input_path,
        conf_threshold=0.25,
        output_path=output_path
    )

    return {
        "task": "Oriented Object Detection",
        "query": query,
        "filename": image.filename,
        "result": result
    }