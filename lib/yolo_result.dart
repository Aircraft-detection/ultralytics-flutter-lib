// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_result.dart

import 'dart:typed_data';
import 'dart:ui';

class YOLOResult {
  
  // modified: add width, height, and landscape
  final int w;
  final int h;
  final bool isLandscape;
  
  final int classIndex;

  final String className;

  final double confidence;

  final Rect boundingBox;

  final Rect normalizedBox;

  YOLOResult({
    // modified: add width, height, and landscape
    required this.w,
    required this.h,
    required this.isLandscape,
    required this.classIndex,
    required this.className,
    required this.confidence,
    required this.boundingBox,
    required this.normalizedBox,
  });

  /// Creates a [YOLOResult] from a map representation.
  ///
  /// This factory constructor is primarily used for deserializing results
  /// received from the platform channel. The map should contain keys:
  /// - 'classIndex': int
  /// - 'className': String
  /// - 'confidence': double
  /// - 'boundingBox': Map with 'left', 'top', 'right', 'bottom'
  /// - 'normalizedBox': Map with 'left', 'top', 'right', 'bottom'
  /// - 'mask': (optional) List of List of double
  /// - 'keypoints': (optional) List of double in x,y,confidence triplets
  factory YOLOResult.fromMap(Map<dynamic, dynamic> map) {
    // modified: add width, height, and landscape
    final w = (map['w'] as num).toInt();
    final h = (map['h'] as num).toInt();
    final isLandscape = map['isLandscape'] as bool;
    
    final classIndex = map['classIndex'] as int;
    final className = map['className'] as String;
    final confidence = (map['confidence'] as num).toDouble();

    // Parse bounding box
    final boxMap = map['boundingBox'] as Map<dynamic, dynamic>;
    final boundingBox = Rect.fromLTRB(
      (boxMap['left'] as num).toDouble(),
      (boxMap['top'] as num).toDouble(),
      (boxMap['right'] as num).toDouble(),
      (boxMap['bottom'] as num).toDouble(),
    );

    // Parse normalized bounding box
    final normalizedBoxMap = map['normalizedBox'] as Map<dynamic, dynamic>;
    final normalizedBox = Rect.fromLTRB(
      (normalizedBoxMap['left'] as num).toDouble(),
      (normalizedBoxMap['top'] as num).toDouble(),
      (normalizedBoxMap['right'] as num).toDouble(),
      (normalizedBoxMap['bottom'] as num).toDouble(),
    );

    // Parse mask if available
    List<List<double>>? mask;
    if (map.containsKey('mask') && map['mask'] != null) {
      final maskData = map['mask'] as List<dynamic>;
      mask = maskData
          .map(
            (row) => (row as List<dynamic>)
                .map((val) => (val as num).toDouble())
                .toList(),
          )
          .toList();
    }

    // Parse keypoints if available
    List<Point>? keypoints;
    List<double>? keypointConfidences;
    if (map.containsKey('keypoints') && map['keypoints'] != null) {
      final keypointsData = map['keypoints'] as List<dynamic>;
      keypoints = [];
      keypointConfidences = [];

      for (var i = 0; i < keypointsData.length; i += 3) {
        keypoints.add(
          Point(
            (keypointsData[i] as num).toDouble(),
            (keypointsData[i + 1] as num).toDouble(),
          ),
        );
        keypointConfidences.add((keypointsData[i + 2] as num).toDouble());
      }
    }

    return YOLOResult(
      // modified: add width, height, and landscape
      w: w,
      h: h,
      isLandscape: isLandscape,
      classIndex: classIndex,
      className: className,
      confidence: confidence,
      boundingBox: boundingBox,
      normalizedBox: normalizedBox,
    );
  }

  /// Converts this [YOLOResult] to a map representation.
  ///
  /// This method is used for serializing the result for platform channel
  /// communication. The returned map contains all the properties of this
  /// result in a format suitable for transmission across platform channels.
  Map<String, dynamic> toMap() {
    final map = <String, dynamic>{
      // modified: add width, height, and landscape
      'w': w,
      'h': h,
      'isLandscape': isLandscape,
      
      'classIndex': classIndex,
      'className': className,
      'confidence': confidence,
      'boundingBox': {
        'left': boundingBox.left,
        'top': boundingBox.top,
        'right': boundingBox.right,
        'bottom': boundingBox.bottom,
      },
      'normalizedBox': {
        'left': normalizedBox.left,
        'top': normalizedBox.top,
        'right': normalizedBox.right,
        'bottom': normalizedBox.bottom,
      },
    };

    return map;
  }

  @override
  String toString() {
    return 'YOLOResult{classIndex: $classIndex, className: $className, confidence: $confidence, boundingBox: $boundingBox}';
  }
}

/// Represents a point in 2D space.
///
/// Used primarily for representing keypoint locations in pose estimation
/// results. Coordinates are typically in pixel space relative to the
/// original image dimensions.
///
/// Example:
/// ```dart
/// final point = Point(150.5, 200.0);
/// print('Point at (${point.x}, ${point.y})');
/// ```
class Point {
  /// The x-coordinate of the point.
  final double x;

  /// The y-coordinate of the point.
  final double y;

  Point(this.x, this.y);

  Map<String, double> toMap() => {'x': x, 'y': y};

  factory Point.fromMap(Map<dynamic, dynamic> map) {
    return Point((map['x'] as num).toDouble(), (map['y'] as num).toDouble());
  }

  @override
  String toString() => 'Point($x, $y)';
}
