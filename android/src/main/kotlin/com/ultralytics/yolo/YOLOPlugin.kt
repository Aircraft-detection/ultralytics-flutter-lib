// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry // Added for RequestPermissionsResultListener
import java.io.ByteArrayOutputStream

class YOLOPlugin : FlutterPlugin, ActivityAware, MethodChannel.MethodCallHandler, PluginRegistry.RequestPermissionsResultListener {

  private lateinit var methodChannel: MethodChannel
  private val instanceChannels = mutableMapOf<String, MethodChannel>()
  private lateinit var applicationContext: android.content.Context
  private var activity: Activity? = null
  private var activityBinding: ActivityPluginBinding? = null // Added to store the binding
  private val TAG = "YOLOPlugin"
  private lateinit var viewFactory: YOLOPlatformViewFactory
  private lateinit var binaryMessenger: io.flutter.plugin.common.BinaryMessenger

  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    // Store application context and binary messenger for later use
    applicationContext = flutterPluginBinding.applicationContext
    binaryMessenger = flutterPluginBinding.binaryMessenger

    // Create and store the view factory for later activity updates
    viewFactory = YOLOPlatformViewFactory(flutterPluginBinding.binaryMessenger)
    
    // Register platform view
    flutterPluginBinding.platformViewRegistry.registerViewFactory(
      "com.ultralytics.yolo/YOLOPlatformView",
      viewFactory
    )

    // Register default method channel for backward compatibility
    methodChannel = MethodChannel(
      flutterPluginBinding.binaryMessenger,
      "yolo_single_image_channel"
    )
    methodChannel.setMethodCallHandler(this)
    
    Log.d(TAG, "YOLOPlugin attached to engine")
  }
  
  override fun onAttachedToActivity(binding: ActivityPluginBinding) {
    activity = binding.activity
    activityBinding = binding // Store the binding
    viewFactory.setActivity(activity)
    activityBinding?.addRequestPermissionsResultListener(this)
    Log.d(TAG, "YOLOPlugin attached to activity: ${activity?.javaClass?.simpleName}, stored binding, and added RequestPermissionsResultListener")
  }

  override fun onDetachedFromActivityForConfigChanges() {
    Log.d(TAG, "YOLOPlugin detached from activity for config changes. Listener will be removed in onDetachedFromActivity.")
    // activity and viewFactory.setActivity(null) will be handled by onDetachedFromActivity
    // activityBinding will also be cleared in onDetachedFromActivity
  }

  override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
    activity = binding.activity
    activityBinding = binding // Store the new binding
    viewFactory.setActivity(activity)
    activityBinding?.addRequestPermissionsResultListener(this) // Add listener with new binding
    Log.d(TAG, "YOLOPlugin reattached to activity: ${activity?.javaClass?.simpleName}, stored new binding, and re-added RequestPermissionsResultListener")
  }

  override fun onDetachedFromActivity() {
    Log.d(TAG, "YOLOPlugin detached from activity")
    activityBinding?.removeRequestPermissionsResultListener(this)
    activityBinding = null
    activity = null
    viewFactory.setActivity(null)
    Log.d(TAG, "Cleared activity, activityBinding, and removed RequestPermissionsResultListener")
  }

  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    methodChannel.setMethodCallHandler(null)
    Log.d(TAG, "YoloPlugin detached from engine")
    // Clean up view factory resources
    viewFactory.dispose()
    // YOLO class doesn't need explicit release
  }
  
  /**
   * Gets the absolute path to the app's internal storage directory
   */
  private fun getInternalStoragePath(): String {
    return applicationContext.filesDir.absolutePath
  }

  /**
   * Resolves a model path that might be relative to app's internal storage
   * @param modelPath The model path from Flutter
   * @return Resolved absolute path or original asset path
   */
  private fun resolveModelPath(modelPath: String): String {
    // If it's already an absolute path, return it
    if (YOLOUtils.isAbsolutePath(modelPath)) {
      return modelPath
    }
    
    // Check if it's a relative path to internal storage
    if (modelPath.startsWith("internal://")) {
      val relativePath = modelPath.substring("internal://".length)
      return "${applicationContext.filesDir.absolutePath}/$relativePath"
    }
    
    // Otherwise, consider it an asset path
    return modelPath
  }

  override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
    when (call.method) {
      "createInstance" -> {
      }
      
      "loadModel" -> {
      }

      "checkModelExists" -> {

      }
      // END OF "checkModelExists" case
      
      "getStoragePaths" -> {
        try {
          val paths = mapOf(
            "internal" to applicationContext.filesDir.absolutePath,
            "cache" to applicationContext.cacheDir.absolutePath,
            "external" to applicationContext.getExternalFilesDir(null)?.absolutePath,
            "externalCache" to applicationContext.externalCacheDir?.absolutePath
          )
          result.success(paths)
        } catch (e: Exception) {
          result.error("path_error", "Failed to get storage paths: ${e.message}", null)
        }
      }
      
      "setModel" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val viewId = args?.get("viewId") as? Int
          val modelPath = args?.get("modelPath") as? String
          val taskString = args?.get("task") as? String
          
          if (viewId == null || modelPath == null || taskString == null) {
            result.error("bad_args", "Missing required arguments for setModel", null)
            return
          }
          
          // Get the YoloView instance from the factory
          val yoloView = viewFactory.activeViews[viewId]
          if (yoloView != null) {
            // Resolve the model path
            val resolvedPath = resolveModelPath(modelPath)
            
            // Call setModel on the YoloView
            yoloView.setModel(resolvedPath) { success ->
              if (success) {
                result.success(null)
              } else {
                result.error("MODEL_NOT_FOUND", "Failed to load model: $modelPath", null)
              }
            }
          } else {
            result.error("VIEW_NOT_FOUND", "YoloView with id $viewId not found", null)
          }
        } catch (e: Exception) {
          Log.e(TAG, "Error setting model", e)
          result.error("set_model_error", "Error setting model: ${e.message}", null)
        }
      }
      
      "disposeInstance" -> {

      }
      
      else -> result.notImplemented()
    }
  }

  // Implementation for PluginRegistry.RequestPermissionsResultListener
  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ): Boolean {
    Log.d(TAG, "onRequestPermissionsResult called in YoloPlugin. requestCode: $requestCode, activeViews: ${viewFactory.activeViews.size}")
    var handled = false
    // Iterate over a copy of the values to avoid concurrent modification issues.
    val viewsToNotify = ArrayList(viewFactory.activeViews.values)
    for (platformView in viewsToNotify) {
        try {
            // YoloPlatformView has the passRequestPermissionsResult method
            platformView.passRequestPermissionsResult(requestCode, permissions, grantResults)
            // YoloPlatformView's passRequestPermissionsResult will log its own viewId
            Log.d(TAG, "Successfully attempted to delegate permission result to an active YoloPlatformView.")
            handled = true
            // Assuming only one view actively requests permissions at a time.
            // If multiple views could request, 'handled' logic might need adjustment
            // or ensure only the correct view processes it.
        } catch (e: Exception) {
            Log.e(TAG, "Error delegating onRequestPermissionsResult to a YoloPlatformView instance", e)
        }
    }
    if (!handled && viewsToNotify.isNotEmpty()) {
        // This log means we iterated views but none seemed to handle it, or an exception occurred.
        Log.w(TAG, "onRequestPermissionsResult was iterated but not confirmed handled by any YoloPlatformView, or an error occurred during delegation.")
    } else if (viewsToNotify.isEmpty()) {
        Log.d(TAG, "onRequestPermissionsResult: No active YoloPlatformViews to notify.")
    }
    return handled // Return true if any view instance successfully processed it.
  }
  
  // Helper function to load labels
  private fun loadLabels(modelPath: String): List<String> {
    // This is a placeholder - in a real implementation, you would load labels from metadata
    return listOf(
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack"
    )
  }
}