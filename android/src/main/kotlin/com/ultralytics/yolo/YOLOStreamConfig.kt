// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

/**
 * Configuration class for YOLOView streaming functionality
 * Controls what data is included in real-time streaming and performance settings
 */
data class YOLOStreamConfig(
    // Original image data (uses ImageProxy bitmap reuse - no additional conversion needed)
    val includeOriginalImage: Boolean = false,
    
    // Performance controls
    val maxFPS: Int? = null,              // Limit inference to max FPS (e.g., 15, 30)
    val throttleIntervalMs: Int? = null,  // Minimum interval between inferences in milliseconds
    
    // Inference frequency controls
    val inferenceFrequency: Int? = null,  // Target inference frequency in FPS (e.g., 5, 10, 15, 30)
    val skipFrames: Int? = null           // Skip frames between inferences (alternative to inferenceFrequency)
    
    // Note: annotatedImage is intentionally excluded for YOLOView
    // YOLOView uses Canvas drawing (real-time overlay), not bitmap generation
) {
}