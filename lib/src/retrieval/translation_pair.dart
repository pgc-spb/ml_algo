/// A translation pair with embedding for creating HybridFTSSearcher.
///
/// Generic naming: source and target can be any languages.
class TranslationPair {
  final String source;
  final String target;
  final List<double> embedding;

  TranslationPair({
    required this.source,
    required this.target,
    required this.embedding,
  });
}

