// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo.dart

import 'dart:async';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_exceptions.dart';
import 'package:ultralytics_yolo/yolo_instance_manager.dart';

/// Exports all YOLO-related classes and enums
export 'yolo_task.dart';
export 'yolo_exceptions.dart';
export 'yolo_result.dart';
export 'yolo_instance_manager.dart';

/// YOLO (You Only Look Once) is a class that provides machine learning inference
/// capabilities for object detection, segmentation, classification, pose estimation,
/// and oriented bounding box detection.
///
/// This class handles the initialization of YOLO models and provides methods
/// to perform inference on images.
///
/// Example usage:
/// ```dart
/// final yolo = YOLO(
///   modelPath: 'assets/models/yolo11n.tflite',
///   task: YOLOTask.detect,
/// );
///
/// await yolo.loadModel();
/// final results = await yolo.predict(imageBytes);
/// ```
class YOLO {
  // Static channel for backward compatibility
  static const _defaultChannel = MethodChannel('yolo_single_image_channel');

  // Instance-specific properties
  late final String _instanceId;
  late final MethodChannel _channel;
  bool _isInitialized = false;

  /// The unique instance ID for this YOLO instance
  String get instanceId => _instanceId;

  /// Path to the YOLO model file. This can be:
  /// - An asset path (e.g., 'assets/models/yolo11n.tflite')
  /// - An absolute file path (e.g., '/data/user/0/com.example.app/files/models/yolo11n.tflite')
  /// - An internal storage reference (e.g., 'internal://models/yolo11n.tflite')
  ///
  /// The 'internal://' prefix will be resolved to the app's internal storage directory.
  final String modelPath;

  /// Classifier options for customizing preprocessing (1-channel support, etc.)
  final Map<String, dynamic>? classifierOptions;

  /// The view ID of the associated YoloView (used for model switching)
  int? _viewId;

  /// Creates a new YOLO instance with the specified model path and task.
  ///
  /// The [modelPath] can refer to a model in assets, internal storage, or absolute path.
  /// The [task] specifies what type of inference will be performed.
  ///
  /// If [useMultiInstance] is true, each YOLO instance gets a unique ID and its own channel.
  /// If false, uses the default channel for backward compatibility.
  YOLO({
    required this.modelPath,
    bool useMultiInstance = false,
    this.classifierOptions,
  }) {
    if (useMultiInstance) {
      // Generate unique instance ID
      _instanceId = 'yolo_${DateTime.now().millisecondsSinceEpoch}_$hashCode';

      // Create instance-specific channel
      final channelName = 'yolo_single_image_channel_$_instanceId';
      _channel = MethodChannel(channelName);

      // Register this instance with the manager
      YOLOInstanceManager.registerInstance(_instanceId, this);
    } else {
      // Use default values for backward compatibility
      _instanceId = 'default';
      _channel = _defaultChannel;
      _isInitialized = true; // Skip initialization for default mode
    }
  }

  /// Sets the view ID for this controller (called internally by YoloView)
  void setViewId(int viewId) {
    _viewId = viewId;
  }

  /// Creates a YOLO instance with classifier options for custom preprocessing
  ///
  /// This constructor is specifically designed for classification models that
  /// need custom preprocessing, such as 1-channel grayscale models.
  ///
  /// Example:
  /// ```dart
  /// final yolo = YOLO.withClassifierOptions(
  ///   modelPath: 'assets/handwriting_model.tflite',
  ///   task: YOLOTask.classify,
  ///   classifierOptions: {
  ///     'enable1ChannelSupport': true,
  ///     'enableColorInversion': true,
  ///     'enableMaxNormalization': true,
  ///     'expectedChannels': 1,
  ///     'expectedClasses': 12,
  ///   },
  /// );
  /// ```
  ///
  /// if need custom Normalization:
  /// ```dart
  ///   final grayscaleOptions = {
  ///   'enableMaxNormalization': false,
  ///   'inputMean': 127.5,
  ///   'inputStd' : 127.5,
  ///   'expectedChannels': 1,
  ///   // labels·expectedClasses (if needed)
  /// };
  ///```

  static YOLO withClassifierOptions({
    required String modelPath,
    required YOLOTask task,
    required Map<String, dynamic> classifierOptions,
    bool useMultiInstance = false,
  }) {
    return YOLO(
      modelPath: modelPath,
      useMultiInstance: useMultiInstance,
      classifierOptions: classifierOptions,
    );
  }

  /// Disposes this YOLO instance and releases all resources
  Future<void> dispose() async {
    try {
      await _channel.invokeMethod('disposeInstance', {
        'instanceId': _instanceId,
      });
    } catch (e) {
      logInfo('Error disposing instance $_instanceId: $e');
    } finally {
      // Always remove from manager, even if platform call fails
      YOLOInstanceManager.unregisterInstance(_instanceId);
      _isInitialized = false;
    }
  }
}
