/// A translation pair with embedding for creating HybridFTSSearcher.
class TranslationPair {
  final String french;
  final String english;
  final List<double> embedding;

  TranslationPair({
    required this.french,
    required this.english,
    required this.embedding,
  });
}

