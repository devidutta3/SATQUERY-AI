/* =========================================================
   SATQUERY AI
   Frontend Controller
========================================================= */


/* =========================================================
   CONFIGURATION
========================================================= */

const API_URL = "http://127.0.0.1:8000";


/* =========================================================
   DOM ELEMENTS
========================================================= */

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const tiffPreview =
    document.getElementById("tiffPreview");

const removeImage =
    document.getElementById("removeImage");

const selectedFileName =
    document.getElementById("selectedFileName");

const selectedFileType =
    document.getElementById("selectedFileType");

const queryInput =
    document.getElementById("queryInput");

const analyzeButton =
    document.getElementById("analyzeButton");

const analysisStatus =
    document.getElementById("analysisStatus");

const taskResult =
    document.getElementById("taskResult");

const answerResult =
    document.getElementById("answerResult");

const confidenceResult =
    document.getElementById("confidenceResult");

const engineResult =
    document.getElementById("engineResult");

const resultPlaceholder =
    document.getElementById("resultPlaceholder");

const resultImagePreview =
    document.getElementById("resultImagePreview");


/* =========================================================
   STATE
========================================================= */

let selectedImage = null;


/* =========================================================
   FILE VALIDATION
========================================================= */

function isValidImageFile(file) {

    if (!file) {
        return false;
    }

    const validExtensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff"
    ];

    const fileName =
        file.name.toLowerCase();

    return validExtensions.some(
        extension => fileName.endsWith(extension)
    );
}


/* =========================================================
   FILE TYPE
========================================================= */

function isTIFF(file) {

    if (!file) {
        return false;
    }

    const name =
        file.name.toLowerCase();

    return (
        name.endsWith(".tif") ||
        name.endsWith(".tiff")
    );
}


/* =========================================================
   HANDLE SELECTED FILE
========================================================= */

function handleSelectedFile(file) {

    if (!file) {
        return;
    }


    if (!isValidImageFile(file)) {

        alert(
            "Unsupported file.\n\n" +
            "Please upload JPG, PNG, WEBP or TIFF."
        );

        return;
    }


    selectedImage = file;


    selectedFileName.textContent =
        file.name;


    selectedFileType.textContent =
        formatFileSize(file.size);


    previewContainer.style.display =
        "block";


    uploadArea.style.display =
        "none";


    /*
     * TIFF files cannot reliably be previewed
     * by the browser as normal images.
     */

    if (isTIFF(file)) {

        imagePreview.style.display =
            "none";

        tiffPreview.style.display =
            "flex";

        selectedFileType.textContent =
            `${formatFileSize(file.size)} • Multispectral GeoTIFF`;

        return;
    }


    /*
     * Normal RGB image preview.
     */

    tiffPreview.style.display =
        "none";

    imagePreview.style.display =
        "block";


    const reader =
        new FileReader();


    reader.onload = function (event) {

        imagePreview.src =
            event.target.result;

    };


    reader.readAsDataURL(file);
}


/* =========================================================
   FORMAT FILE SIZE
========================================================= */

function formatFileSize(bytes) {

    if (bytes === 0) {
        return "0 Bytes";
    }

    const units = [
        "Bytes",
        "KB",
        "MB",
        "GB"
    ];

    const index =
        Math.floor(
            Math.log(bytes) /
            Math.log(1024)
        );

    return (
        parseFloat(
            (bytes /
                Math.pow(1024, index)
            ).toFixed(2)
        ) +
        " " +
        units[index]
    );
}


/* =========================================================
   FILE INPUT
========================================================= */

imageInput.addEventListener(
    "change",
    function () {

        const file =
            this.files[0];

        handleSelectedFile(file);
    }
);


/* =========================================================
   DRAG & DROP
========================================================= */

uploadArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadArea.classList.add(
            "dragging"
        );
    }
);


uploadArea.addEventListener(
    "dragleave",
    function () {

        uploadArea.classList.remove(
            "dragging"
        );
    }
);


uploadArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadArea.classList.remove(
            "dragging"
        );


        const files =
            event.dataTransfer.files;


        if (!files || files.length === 0) {
            return;
        }


        handleSelectedFile(files[0]);
    }
);


/* =========================================================
   CLICK UPLOAD AREA
========================================================= */

uploadArea.addEventListener(
    "click",
    function (event) {

        /*
         * Don't trigger the file picker twice
         * when the actual label/button is clicked.
         */

        if (
            event.target.closest(
                ".upload-button"
            )
        ) {
            return;
        }


        imageInput.click();
    }
);


/* =========================================================
   REMOVE IMAGE
========================================================= */

removeImage.addEventListener(
    "click",
    function () {

        selectedImage = null;

        imageInput.value = "";

        imagePreview.src = "";

        imagePreview.style.display =
            "none";

        tiffPreview.style.display =
            "none";

        previewContainer.style.display =
            "none";

        uploadArea.style.display =
            "flex";


        resetResults();
    }
);


/* =========================================================
   QUICK QUERY BUTTONS
========================================================= */

const suggestionButtons =
    document.querySelectorAll(
        ".suggestion"
    );


suggestionButtons.forEach(
    function (button) {

        button.addEventListener(
            "click",
            function () {

                /*
                 * Remove the visual
                 * quick-query prefix.
                 */

                let text =
                    button.textContent
                    .replace("◇", "")
                    .trim();


                queryInput.value =
                    text;


                queryInput.focus();
            }
        );
    }
);


/* =========================================================
   RESET RESULTS
========================================================= */

function resetResults() {

    analysisStatus.textContent =
        "WAITING";

    taskResult.textContent =
        "—";

    answerResult.textContent =
        "Upload an image and ask a question to begin analysis.";

    confidenceResult.textContent =
        "—";

    engineResult.textContent =
        "—";


    resultPlaceholder.style.display =
        "flex";

    resultImagePreview.style.display =
        "none";

    resultImagePreview.src = "";
}


/* =========================================================
   STATUS
========================================================= */

function setAnalysisStatus(
    status
) {

    analysisStatus.textContent =
        status;
}


/* =========================================================
   RESULT IMAGE URL
========================================================= */

function getOutputImageURL(path) {

    if (!path) {
        return null;
    }


    /*
     * Backend currently returns
     * Windows filesystem paths.
     *
     * We only need the output filename
     * because FastAPI serves /outputs/.
     */

    const filename =
        path
            .split("\\")
            .pop()
            .split("/")
            .pop();


    return (
        `${API_URL}/outputs/` +
        encodeURIComponent(filename)
    );
}


/* =========================================================
   SHOW RESULT IMAGE
========================================================= */

function showResultImage(path) {

    const url =
        getOutputImageURL(path);


    if (!url) {
        return;
    }


    resultImagePreview.onload =
        function () {

            resultPlaceholder.style.display =
                "none";

            resultImagePreview.style.display =
                "block";
        };


    resultImagePreview.onerror =
        function () {

            resultPlaceholder.style.display =
                "flex";

            resultImagePreview.style.display =
                "none";

            console.warn(
                "Could not load result image:",
                url
            );
        };


    resultImagePreview.src =
        url;
}


/* =========================================================
   ANALYZE IMAGE
========================================================= */

async function analyzeImage() {


    /* ---------- Check image ---------- */

    if (!selectedImage) {

        alert(
            "Please upload a satellite image first."
        );

        return;
    }


    /* ---------- Check query ---------- */

    const query =
        queryInput.value.trim();


    if (!query) {

        alert(
            "Please enter a query."
        );

        queryInput.focus();

        return;
    }


    /* ---------- UI loading ---------- */

    analyzeButton.disabled =
        true;


    analyzeButton.querySelector(
        "span"
    ).textContent =
        "Analyzing...";


    setAnalysisStatus(
        "PROCESSING"
    );


    taskResult.textContent =
        "Analyzing satellite observation...";


    answerResult.textContent =
        "Running specialist AI model...";


    confidenceResult.textContent =
        "—";


    engineResult.textContent =
        "Routing...";


    resultPlaceholder.style.display =
        "flex";


    resultImagePreview.style.display =
        "none";


    resultImagePreview.src = "";


    /* ---------- Form data ---------- */

    const formData =
        new FormData();


    formData.append(
        "image",
        selectedImage
    );


    formData.append(
        "query",
        query
    );


    try {


        /* ---------- API request ---------- */

        const response =
            await fetch(
                `${API_URL}/analyze`,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data =
            await response.json();


        console.log(
            "SatQuery response:",
            data
        );


        /* ---------- Backend error ---------- */

        if (
            data.result &&
            data.result.error
        ) {

            setAnalysisStatus(
                "ERROR"
            );


            taskResult.textContent =
                data.task ||
                "Analysis Error";


            answerResult.textContent =
                data.result.error;


            confidenceResult.textContent =
                "—";


            engineResult.textContent =
                "Prithvi";


            return;
        }


        /* ---------- Task ---------- */

        taskResult.textContent =
            data.task ||
            "Analysis";


        const result =
            data.result || {};


        /* =================================================
           YOLO-OBB
        ================================================= */

        if (
            result.model ===
            "YOLO-OBB"
        ) {

            setAnalysisStatus(
                "COMPLETE"
            );


            engineResult.textContent =
                "YOLO-OBB";


            const detections =
                result.detections || [];


            const count =
                result.num_objects ??
                detections.length;


            /*
             * No objects
             */

            if (count === 0) {

                answerResult.textContent =
                    "No objects were detected in this satellite image.";

                confidenceResult.textContent =
                    "—";

            }


            /*
             * Objects detected
             */

            else {

                const classes =
                    detections.map(
                        detection =>
                            detection.class_name
                    );


                const uniqueClasses =
                    [
                        ...new Set(classes)
                    ];


                const averageConfidence =
                    detections.length > 0
                        ?
                        detections.reduce(
                            (
                                total,
                                detection
                            ) =>
                                total +
                                detection.confidence,
                            0
                        ) /
                        detections.length
                        :
                        0;


                const percentage =
                    (
                        averageConfidence *
                        100
                    ).toFixed(1);


                answerResult.textContent =
                    `${count} object${count === 1 ? "" : "s"} detected: ${uniqueClasses.join(", ")}.`;


                confidenceResult.textContent =
                    `${percentage}%`;
            }


            /*
             * Show YOLO visualization
             */

            if (
                result.output_image
            ) {

                showResultImage(
                    result.output_image
                );
            }

        }


        /* =================================================
           PRITHVI BURN SCAR SEGMENTATION
        ================================================= */

        else if (
            result.model ===
            "Prithvi-EO-2.0-300M-BurnScars"
        ) {

            setAnalysisStatus(
                "COMPLETE"
            );


            engineResult.textContent =
                "Prithvi-EO-2.0";


            const burnedArea =
                Number(
                    result.burned_area_percentage ||
                    0
                );


            const meanProbability =
                Number(
                    result.mean_burn_probability ||
                    0
                );


            const probabilityPercent =
                (
                    meanProbability *
                    100
                ).toFixed(1);


            /*
             * Main answer
             */

            answerResult.textContent =
                `Burn scars cover approximately ${burnedArea.toFixed(2)}% of the analyzed scene.`;


            /*
             * Confidence-like model
             * probability metric
             */

            confidenceResult.textContent =
                `${probabilityPercent}%`;


            /*
             * Show segmentation overlay
             */

            if (
                result.output_overlay
            ) {

                showResultImage(
                    result.output_overlay
                );

            }

        }


        /* =================================================
           UNKNOWN MODEL
        ================================================= */

        else {

            setAnalysisStatus(
                "COMPLETE"
            );


            engineResult.textContent =
                result.model ||
                "AI";


            answerResult.textContent =
                "The satellite image was successfully analyzed.";

        }


    }


    /* =====================================================
       ERROR HANDLING
    ===================================================== */

    catch (error) {

        console.error(
            "SatQuery error:",
            error
        );


        setAnalysisStatus(
            "ERROR"
        );


        taskResult.textContent =
            "Analysis failed";


        answerResult.textContent =
            "Unable to connect to the SatQuery AI backend. Make sure FastAPI is running on port 8000.";


        confidenceResult.textContent =
            "—";


        engineResult.textContent =
            "Offline";


        resultPlaceholder.style.display =
            "flex";


        resultImagePreview.style.display =
            "none";


        alert(
            "Could not connect to SatQuery AI backend.\n\n" +
            "Please check that FastAPI is running."
        );

    }


    /* =====================================================
       RESTORE BUTTON
    ===================================================== */

    finally {

        analyzeButton.disabled =
            false;


        analyzeButton.querySelector(
            "span"
        ).textContent =
            "Analyze Image";
    }

}


/* =========================================================
   ANALYZE BUTTON EVENT
========================================================= */

analyzeButton.addEventListener(
    "click",
    analyzeImage
);


/* =========================================================
   KEYBOARD SHORTCUT
========================================================= */

queryInput.addEventListener(
    "keydown",
    function (event) {

        /*
         * Ctrl + Enter
         * or
         * Cmd + Enter
         */

        if (
            event.key === "Enter" &&
            (event.ctrlKey ||
             event.metaKey)
        ) {

            event.preventDefault();

            analyzeImage();
        }
    }
);


/* =========================================================
   INITIAL STATE
========================================================= */

resetResults();


console.log(
    "🛰️ SatQuery AI frontend initialized."
);