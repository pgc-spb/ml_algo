/// Result of a translation search query.
class TranslationResult {
  /// The point index in the searcher.
  final int pointIndex;
  
  /// The source text (can be any language).
  final String sourceText;
  
  /// The target text (can be any language).
  final String targetText;
  
  /// The semantic distance (lower is more similar).
  final double distance;
  
  TranslationResult({
    required this.pointIndex,
    required this.sourceText,
    required this.targetText,
    required this.distance,
  });
}

