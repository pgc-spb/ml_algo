/// Result of a translation search query.
class TranslationResult {
  /// The point index in the searcher.
  final int pointIndex;
  
  /// The French text.
  final String frenchText;
  
  /// The English text.
  final String englishText;
  
  /// The semantic distance (lower is more similar).
  final double distance;
  
  TranslationResult({
    required this.pointIndex,
    required this.frenchText,
    required this.englishText,
    required this.distance,
  });
}

