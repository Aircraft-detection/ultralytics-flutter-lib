// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../../models/model_type.dart';
import '../../models/slider_type.dart';
import '../../services/model_manager.dart';

class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  ModelType _selectedModel = ModelType.detect;
  String? _modelPath;

  final _yoloController = YOLOViewController();
  final _yoloViewKey = GlobalKey<YOLOViewState>();
  final bool _useController = true;

  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();

    // Initialize ModelManager
    _modelManager = ModelManager();

    // Load initial model
    _loadModelForPlatform();

    // Set initial thresholds after frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_useController) {
        _yoloController.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      } else {
        _yoloViewKey.currentState?.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      }
    });
  }

  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;

    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;

    if (elapsed >= 1000) {
      final calculatedFps = _frameCount * 1000 / elapsed;
      debugPrint('Calculated FPS: ${calculatedFps.toStringAsFixed(1)}');

      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    // Debug first few detections
    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        'Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final orientation = MediaQuery.of(context).orientation;

    return Scaffold(
      body: Stack(
        children: [
          // YOLO View: must be at back
          if (_modelPath != null)
            YOLOView(
              key: _useController
                  ? const ValueKey('yolo_view_static')
                  : _yoloViewKey,
              controller: _useController ? _yoloController : null,
              modelPath: _modelPath!,
              task: _selectedModel.task,
              onResult: _onDetectionResults,
            )
          else
            const Center(
              child: Text(
                'No model loaded',
                style: TextStyle(color: Colors.white),
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _loadModelForPlatform() async {
    setState(() {
      _frameCount = 0;
      _lastFpsUpdate = DateTime.now();
    });

    try {
      // Use ModelManager to get the model path
      // This will automatically download if not found locally
      final modelPath = await _modelManager.getModelPath(_selectedModel);

      if (mounted) {
        setState(() {
          _modelPath = modelPath;
        });

        if (modelPath != null) {
          debugPrint('CameraInferenceScreen: Model path set to: $modelPath');
        } else {
          // Model loading failed
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: const Text('Model Not Available'),
              content: Text(
                'Failed to load ${_selectedModel.modelName} model. Please check your internet connection and try again.',
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('OK'),
                ),
              ],
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Error loading model: $e');
      if (mounted) {
        // Show error dialog
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Model Loading Error'),
            content: Text(
              'Failed to load ${_selectedModel.modelName} model: ${e.toString()}',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
    }
  }
}
