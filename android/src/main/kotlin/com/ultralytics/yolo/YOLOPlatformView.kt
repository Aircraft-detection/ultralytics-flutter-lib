// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.util.Log
import android.view.View
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView

/**
 * YOLOPlatformView - Native view bridge from Flutter
 */
class YOLOPlatformView(
    private val context: Context,
    private val viewId: Int,
    creationParams: Map<String?, Any?>?,
    private val streamHandler: EventChannel.StreamHandler,
    private val methodChannel: MethodChannel?,
    private val factory: YOLOPlatformViewFactory // Added factory reference
) : PlatformView, MethodChannel.MethodCallHandler {

    private val yoloView: YOLOView = YOLOView(context)
    private val TAG = "YOLOPlatformView"
    
    // Initialization flag
    private var initialized = false
    
    init {
        Log.d(TAG, "YOLOPlatformView[$viewId init]: Initialized with creationParams: $creationParams.")

        // Parse model path and task from creation params
        var modelPath = creationParams?.get("modelPath") as? String ?: "yolo11n"

        // Set up the method channel handler
        methodChannel?.setMethodCallHandler(this)
        
        // Set up streaming callback to forward data to Flutter via event channel
        yoloView.setStreamCallback { streamData ->
            // Forward streaming data from YOLOView to Flutter
            sendStreamDataToFlutter(streamData)
        }

        // Attempt to initialize camera as soon as the view is created.
        // YOLOView.initCamera() handles permissions and starts the camera preview.
        Log.d(TAG, "Attempting early camera initialization for YOLOView.")
        yoloView.initCamera() // This will attempt to start camera or request permissions

        // If context is already a LifecycleOwner, inform YOLOView immediately
        if (context is LifecycleOwner) {
            Log.d(TAG, "Initial context is a LifecycleOwner (${context.javaClass.simpleName}), notifying YOLOView.")
            yoloView.onLifecycleOwnerAvailable(context)
        } else {
            Log.w(TAG, "Initial context (${context.javaClass.simpleName}) is NOT a LifecycleOwner. YOLOView will wait for one to be provided via notifyLifecycleOwnerAvailable.")
        }
        
        try {
            Log.d(TAG, "Initializing YOLOPlatformView with model: $modelPath, viewId: $viewId")
            
            // Set up callback for model loading result
            yoloView.setOnModelLoadCallback { success ->
                if (success) {
                    Log.d(TAG, "Model loaded successfully: $modelPath.")
                    // Camera initialization was already attempted.
                    // Mark that the full initialization sequence (including model load) is complete.
                    initialized = true
                } else {
                    Log.w(TAG, "Failed to load model: $modelPath. Camera will run without inference.")
                    // Still mark as initialized since camera can work without model
                    initialized = true
                }
            }
            
            // YOLOView streaming is now configured separately
            // Keep simple inference callback for compatibility
            yoloView.setOnInferenceCallback { result ->
            }
            
            // Load model with the specified path and task
            yoloView.setModel(modelPath)
            
            // Setup zoom callback
            yoloView.onZoomChanged = { zoomLevel ->
                methodChannel?.invokeMethod("onZoomChanged", zoomLevel.toDouble())
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YOLOPlatformView", e)
        }
    }
    
    // Handle method calls from Flutter
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
    }
    
    /**
     * Send stream data to Flutter via event channel
     */
    private fun sendStreamDataToFlutter(streamData: Map<String, Any>) {
        try {
            
            // Create a runnable to ensure we're on the main thread
            val sendResults = Runnable {
                try {
                    if (streamHandler is CustomStreamHandler) {
                        val customHandler = streamHandler as CustomStreamHandler
                        
                        // Use the safe send method
                        val sent = customHandler.safelySend(streamData)
                        if (sent) {
                        } else {
                            Log.w(TAG, "Failed to send stream data via CustomStreamHandler")
                            // Notify Flutter to recreate the channel
                            methodChannel?.invokeMethod("recreateEventChannel", null)
                        }
                    } else {
                        // Use reflection to access the sink property regardless of exact type
                        Log.d(TAG, "Attempting to access sink via reflection")
                        val fields = streamHandler.javaClass.declaredFields
                        Log.d(TAG, "Available fields: ${fields.joinToString { it.name }}")
                        
                        val sinkField = streamHandler.javaClass.getDeclaredField("sink")
                        sinkField.isAccessible = true
                        val sink = sinkField.get(streamHandler) as? EventChannel.EventSink
                        
                        if (sink != null) {
                            sink.success(streamData)
                        } else {
                            Log.w(TAG, "Event sink is NOT available via reflection, skipping data")
                            // Try alternative approach - recreate the event channel
                            Log.d(TAG, "Requesting Flutter to recreate event channel")
                            methodChannel?.invokeMethod("recreateEventChannel", null)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error sending stream data on main thread", e)
                    e.printStackTrace()
                }
            }
            
            // Make sure we're on the main thread when sending events
            val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
            mainHandler.post(sendResults)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing stream data", e)
            e.printStackTrace()
        }
    }

    override fun getView(): View {
        Log.d(TAG, "Getting view: ${yoloView.javaClass.simpleName}")
        
        // Check if context is a LifecycleOwner
        if (context is androidx.lifecycle.LifecycleOwner) {
            val lifecycleOwner = context as androidx.lifecycle.LifecycleOwner
            Log.d(TAG, "Context is a LifecycleOwner with state: ${lifecycleOwner.lifecycle.currentState}")
        } else {
            Log.e(TAG, "Context is NOT a LifecycleOwner! This may cause camera issues.")
        }
        
        // Try setting custom layout parameters
        if (yoloView.layoutParams == null) {
            yoloView.layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT
            )
            Log.d(TAG, "Set layout params for YOLOView")
        }
        
        return yoloView
    }

    override fun dispose() {
        Log.d(TAG, "Disposing YOLOPlatformView for viewId: $viewId")

        try {
            // Stop camera and inference before disposing
            Log.d(TAG, "Calling yoloView.stop() to stop camera and inference")
            yoloView.stop()

            // Clean up method channel
            Log.d(TAG, "Clearing method channel handler")
            methodChannel?.setMethodCallHandler(null)

            // Notify the factory that this view is disposed
            Log.d(TAG, "Notifying factory of disposal")
            factory.onPlatformViewDisposed(viewId)

            Log.d(TAG, "YOLOPlatformView disposal completed successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Error during YOLOPlatformView disposal", e)
        }
    }

    /**
     * Called by YOLOPlatformViewFactory when the Activity (which is a LifecycleOwner)
     * becomes available or changes.
     */
    fun notifyLifecycleOwnerAvailable(owner: LifecycleOwner) {
        Log.d(TAG, "LifecycleOwner (${owner.javaClass.simpleName}) is now available for viewId: $viewId. Notifying YOLOView.")
        yoloView.onLifecycleOwnerAvailable(owner)
    }
        
    // Called by YOLOPlugin to delegate permission results
    fun passRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>, 
        grantResults: IntArray
    ) {
        Log.d(TAG, "passRequestPermissionsResult called in YOLOPlatformView for viewId $viewId, delegating to yoloView")
        yoloView.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
}
