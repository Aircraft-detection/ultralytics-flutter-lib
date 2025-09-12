// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.metadata.schema.ModelMetadata
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.content.res.AssetManager
import android.graphics.Matrix

import org.json.JSONObject

import java.io.ByteArrayInputStream
import java.nio.MappedByteBuffer
import java.nio.charset.StandardCharsets

import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

/**
 * High-performance ObjectDetector that assumes no 90-degree rotation is needed
 * - Performs "resize -> getPixels -> ByteBuffer" in one pass, minimizing Canvas drawing
 * - Reuses Bitmap / ByteBuffer to reduce allocations
 * - Reuses inference output arrays
 */
class ObjectDetector(
    context: Context,
    modelPath: String,
    var labels: List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ),
) {

    var isUpdating: Boolean = false
    protected lateinit var interpreter: Interpreter
    lateinit var inputSize: Size
    protected lateinit var modelInputSize: Pair<Int, Int>
    protected fun isInterpreterInitialized() = this::interpreter.isInitialized

    protected var t0: Long = 0L
    protected var t2: Double = 0.0
    protected var t3: Long = System.nanoTime()
    protected var t4: Double = 0.0

    var CONFIDENCE_THRESHOLD: Float = 0.25f
    var IOU_THRESHOLD: Float = 0.4f
    var transformationMatrix: Matrix? = null
    var pendingBitmapFrame: Bitmap? = null
    var isFrontCamera: Boolean = false

    protected fun updateTiming() {
        val now = System.nanoTime()
        val dt = (now - t0) / 1e9
        t2 = 0.05 * dt + 0.95 * t2
        t4 = 0.05 * ((now - t3) / 1e9) + 0.95 * t4
        t3 = now
    }

    // Inference output dimensions
    private var out1 = 0
    private var out2 = 0

    // Three image processors: camera portrait, camera landscape, and single images
    private lateinit var imageProcessorCameraPortrait: ImageProcessor
    private lateinit var imageProcessorCameraLandscape: ImageProcessor

    // Reuse inference output array ([1][out1][out2])
    private lateinit var rawOutput: Array<Array<FloatArray>>

    // Transposed array for post-processing
    private lateinit var predictions: Array<FloatArray>

    // ======== Workspace for fast preprocessing ========
    // (1) Temporary scaled Bitmap matching model input size
    //     No 90-degree rotation needed, so simply cache createScaledBitmap() equivalent
    private lateinit var scaledBitmap: Bitmap

    // (2) Array to temporarily store pixels (inWidth*inHeight)
    private lateinit var intValues: IntArray

    // (3) ByteBuffer for TFLite input (1 * height * width * 3 * 4 bytes)
    private lateinit var inputBuffer: ByteBuffer

    // ========== TFLite Interpreter ==========
    // Use protected var interpreter: Interpreter? = null from BasePredictor if available
    // Otherwise, keep it in this class as usual
    init {
        // load model from assets
        val modelBuffer = try {
            Log.d(TAG, "Loading model from assets: $modelPath")
            FileUtil.loadMappedFile(context, modelPath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model with path: $modelPath, error: ${e.message}")
            throw e
        }
        
        val interpreterOptions = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors())

            try {
                addDelegate(GpuDelegate())
                Log.d("ObjectDetector", "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e("ObjectDetector", "GPU delegate error: ${e.message}")
            }
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)
        
        // Call allocateTensors() once during initialization, not in the inference loop
        interpreter.allocateTensors()
        Log.d("TAG", "TFLite model loaded: $modelPath, tensors allocated")

        // Check input shape (example: [1, inHeight, inWidth, 3])
        val inputShape = interpreter.getInputTensor(0).shape()
        val inBatch = inputShape[0]         // Usually 1
        val inHeight = inputShape[1]        // Example: 320
        val inWidth = inputShape[2]         // Example: 320
        val inChannels = inputShape[3]      // 3 (RGB)
        require(inBatch == 1 && inChannels == 3) {
            "Input tensor shape not supported. Expected [1, H, W, 3]. But got ${inputShape.joinToString()}"
        }
        inputSize = Size(inWidth, inHeight) // Set variable in BasePredictor
        modelInputSize = Pair(inWidth, inHeight)
        Log.d("TAG", "Model input size = $inWidth x $inHeight")

        // Output shape (varies by model, modify as needed)
        // Example: [1, 84, 2100] = [batch, outHeight, outWidth]
        val outputShape = interpreter.getOutputTensor(0).shape()
        out1 = outputShape[1] // 84
        out2 = outputShape[2] // 2100
        Log.d("TAG", "Model output shape = [1, $out1, $out2]")

        // Allocate preprocessing resources
        initPreprocessingResources(inWidth, inHeight)

        // Allocate inference output arrays
        rawOutput = Array(1) { Array(out1) { FloatArray(out2) } }
        predictions = Array(out2) { FloatArray(out1) }
        
        // For camera feed in portrait mode - includes 270-degree rotation
        imageProcessorCameraPortrait = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 270-degree rotation (3 * 90 degrees) for back camera
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
        
        // For camera feed in landscape mode - no rotation needed
        imageProcessorCameraLandscape = ImageProcessor.Builder()
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()

        Log.d("TAG", "ObjectDetector initialized.")
    }

    private fun initPreprocessingResources(width: Int, height: Int) {
        // ARGB_8888 Bitmap for input size (e.g., 320x320)
        scaledBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Int array for pixel reading
        intValues = IntArray(width * height)

        // Buffer for TFLite input
        inputBuffer = ByteBuffer.allocateDirect(1 * width * height * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    /**
     * Main inference method
     * - Preprocessing: resize bitmap (scaledBitmap) → getPixels → inputBuffer
     * - TFLite run
     * - Postprocessing (NMS via JNI, etc.)
     * @param bitmap Input bitmap to process
     * @param origWidth Original width of the source image
     * @param origHeight Original height of the source image
     * @param rotateForCamera Whether this is a camera feed that requires rotation (true) or a single image (false)
     * @return YOLOResult containing detection results
     */
    fun predict(
        bitmap: Bitmap,
        origWidth: Int,
        origHeight: Int,
        isLandscape: Boolean
    ): YOLOResult {
        val overallStartTime = System.nanoTime()
        var stageStartTime = System.nanoTime()

        // ======== Preprocessing: Convert Bitmap to ByteBuffer via TensorImage ========
        Log.d(TAG, "Predict Start: Preprocessing")
        // 1. Resize to input size (using createScaledBitmap instead of the original scaledBitmap)
        // val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize.width, inputSize.height, false)

        // 2. Load into TensorImage - reuse tensorImage if possible
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        // 3. Normalization & casting via ImageProcessor (equivalent to [pixel/255])
        // Apply rotation for camera frames, process without rotation for single images
        // Clear inputBuffer before reuse to avoid memory leaks
        inputBuffer.clear()

        val processedImage = if (isLandscape) {
            imageProcessorCameraLandscape.process(tensorImage)
        } else {
            imageProcessorCameraPortrait.process(tensorImage)
        }

        // Reuse our direct ByteBuffer instead of the processedImage.buffer
        inputBuffer.put(processedImage.buffer)
        inputBuffer.rewind()

        var preprocessTimeMs = (System.nanoTime() - stageStartTime) / 1_000_000.0
        Log.d(TAG, "Predict Stage: Preprocessing done in $preprocessTimeMs ms")
        stageStartTime = System.nanoTime()

        // ======== Inference ============
        Log.d(TAG, "Predict Start: Inference")
        interpreter.run(inputBuffer, rawOutput)
        var inferenceTimeMs = (System.nanoTime() - stageStartTime) / 1_000_000.0
        Log.d(TAG, "Predict Stage: Inference done in $inferenceTimeMs ms")
        stageStartTime = System.nanoTime()

        // ======== Post-processing (same as existing code) ============
        Log.d(TAG, "Predict Start: Postprocessing")
        // val postStart = System.nanoTime() // This was previously here, now using stageStartTime
        val outHeight = rawOutput[0].size      // out1
        val outWidth = rawOutput[0][0].size      // out2
        val shape = interpreter.getOutputTensor(0).shape() // example: [1, 84, 8400]
        Log.d("TFLite", "Output shape: " + shape.contentToString())

        val resultBoxes = postprocess(
            rawOutput[0],
            w = outWidth,   // width is out2
            h = outHeight,  // height is out1
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold,
            numItemsThreshold = numItemsThreshold,
            numClasses = labels.size
        )
        for ((index, boxArray) in resultBoxes.withIndex()) {
            Log.d(TAG, "Postprocess result - Box $index: ${boxArray.joinToString(", ")}")
        }
        // Convert to Box list
        val boxes = mutableListOf<Box>()
        for (boxArray in resultBoxes) {
            if (boxArray.size >= 6) {
                // Create xywh (absolute pixel coordinates)
                val rect = RectF(
                    boxArray[0] * origWidth,                    // x
                    boxArray[1] * origHeight,                   // y
                    (boxArray[0] + boxArray[2]) * origWidth,    // right
                    (boxArray[1] + boxArray[3]) * origHeight    // bottom
                )

                // Create xywhn (normalized coordinates 0-1)
                val normRect = RectF(
                    boxArray[0],                    // normalized x
                    boxArray[1],                    // normalized y
                    boxArray[0] + boxArray[2],      // normalized right
                    boxArray[1] + boxArray[3]       // normalized bottom
                )

                // Ensure coordinates are valid
                if (rect.left >= 0 && rect.top >= 0 &&
                    rect.right <= origWidth && rect.bottom <= origHeight &&
                    rect.width() > 0 && rect.height() > 0
                ) {

                    val classIdx = boxArray[5].toInt()
                    val label = if (classIdx in labels.indices) labels[classIdx] else "Unknown"
                    boxes.add(Box(classIdx, label, boxArray[4], rect, normRect))
                }
            }
        }

        // val postEnd = System.nanoTime() // This was previously here, now using stageStartTime for end of postprocess
        var postprocessTimeMs = (System.nanoTime() - stageStartTime) / 1_000_000.0
        Log.d(TAG, "Predict Stage: Postprocessing done in $postprocessTimeMs ms")

        val totalMs = (System.nanoTime() - overallStartTime) / 1_000_000.0
        Log.d(
            TAG,
            "Predict Total time: $totalMs ms (Pre: $preprocessTimeMs, Inf: $inferenceTimeMs, Post: $postprocessTimeMs)"
        )

        updateTiming() // This updates t0, t1, t2, t3, t4 based on its own logic

        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            speed = totalMs, // Actual processing time in milliseconds for this frame
            fps = if (t4 > 0.0) 1.0 / t4 else 0.0, // Smoothed FPS from BasePredictor (t4 is smoothed dt)
            names = labels
        )
    }

    // Thresholds (like setConfidenceThreshold, setIouThreshold in TFLiteDetector)
    // modified: set thresholds
    private var confidenceThreshold = 0.35f
    private var iouThreshold = 0.4f
    private var numItemsThreshold = 30

    // Post-processing via JNI
    private external fun postprocess(
        predictions: Array<FloatArray>,
        w: Int,
        h: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        numItemsThreshold: Int,
        numClasses: Int
    ): Array<FloatArray>

    companion object {
        private const val TAG = "ObjectDetector"

        // Load JNI library
        init {
            System.loadLibrary("ultralytics")
        }

        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
    }
}