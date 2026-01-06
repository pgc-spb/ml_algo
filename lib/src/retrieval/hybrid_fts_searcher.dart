import 'package:ml_algo/src/persistence/neighbor_search_store.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher.dart';
import 'package:ml_algo/src/retrieval/translation_pair.dart';
import 'package:ml_algo/src/retrieval/translation_result.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

/// Hybrid searcher that combines SQLite Full-Text Search (FTS) with
/// Random Binary Projection Search (RBPS) for translation tasks.
///
/// **Simplest Usage:**
/// ```dart
/// // Create from translation pairs (one line!)
/// final searcher = await HybridFTSSearcher.createFromTranslations(
///   store,
///   [
///     TranslationPair(french: 'Bonjour', english: 'Hello', embedding: [...]),
///     // ... more pairs
///   ],
/// );
///
/// // Search (three simple methods)
/// final results = await searcher.searchHybrid(
///   keyword: 'bonjour',
///   embedding: queryEmbedding,
///   k: 10,
/// );
/// ```
///
/// This searcher supports:
/// - **Keyword-only search** (fast FTS): `searchByKeyword()`
/// - **Semantic-only search** (RBPS): `searchBySemantic()`
/// - **Hybrid search** (recommended): `searchHybrid()` - keyword filtering + semantic ranking
class HybridFTSSearcher {
  final SQLiteNeighborSearchStore _store;
  final String _searcherId;
  final RandomBinaryProjectionSearcher _rbpsSearcher;

  HybridFTSSearcher._(
    this._store,
    this._searcherId,
    this._rbpsSearcher,
  );

  /// Creates a HybridFTSSearcher from translation pairs.
  ///
  /// This is the simplest way to create a searcher - just provide your
  /// translation pairs with embeddings and it handles everything.
  ///
  /// Example:
  /// ```dart
  /// final translations = [
  ///   TranslationPair(source: 'Bonjour', target: 'Hello', embedding: [0.1, 0.2, 0.3]),
  ///   TranslationPair(source: 'Au revoir', target: 'Goodbye', embedding: [0.4, 0.5, 0.6]),
  /// ];
  ///
  /// final searcher = await HybridFTSSearcher.createFromTranslations(
  ///   store,
  ///   translations,
  ///   digitCapacity: 8,
  ///   searcherId: 'my-translations',
  /// );
  /// ```
  static Future<HybridFTSSearcher> createFromTranslations(
    SQLiteNeighborSearchStore store,
    List<TranslationPair> translations, {
    int digitCapacity = 8,
    int? seed,
    String? searcherId,
    DType dtype = DType.float32,
  }) async {
    if (translations.isEmpty) {
      throw ArgumentError('Translations list cannot be empty');
    }

    // Validate all translations
    final embeddingDim = translations.first.embedding.length;
    for (var i = 0; i < translations.length; i++) {
      final t = translations[i];
      if (t.source.trim().isEmpty) {
        throw ArgumentError('Translation at index $i has empty source text');
      }
      if (t.target.trim().isEmpty) {
        throw ArgumentError('Translation at index $i has empty target text');
      }
      if (t.embedding.length != embeddingDim) {
        throw ArgumentError(
          'Translation at index $i has embedding dimension ${t.embedding.length}, '
          'expected $embeddingDim',
        );
      }
    }

    // Extract embeddings
    final embeddings = translations.map((t) => t.embedding).toList();
    final data = DataFrame(embeddings, headerExists: false);

    // Create and save searcher
    final searcher = RandomBinaryProjectionSearcher(
      data,
      digitCapacity,
      seed: seed,
      dtype: dtype,
    );

    final savedId = searcherId != null
        ? await searcher.saveToStore(store, searcherId: searcherId)
        : await searcher.saveToStore(store);

    // Batch add text content (more efficient than one-by-one)
    await _batchAddTextContent(store, savedId, translations);

    // Load and return HybridFTSSearcher
    return await loadFromStore(store, savedId);
  }

  /// Batch adds text content for all translations.
  static Future<void> _batchAddTextContent(
    SQLiteNeighborSearchStore store,
    String searcherId,
    List<TranslationPair> translations,
  ) async {
    // Add text content for all translations
    for (var i = 0; i < translations.length; i++) {
      await store.addTextContent(
        searcherId,
        i,
        sourceText: translations[i].source.trim(),
        targetText: translations[i].target.trim(),
      );
    }
  }

  /// Loads a HybridFTSSearcher from a store.
  static Future<HybridFTSSearcher> loadFromStore(
    NeighborSearchStore store,
    String searcherId,
  ) async {
    if (store is! SQLiteNeighborSearchStore) {
      throw ArgumentError(
        'HybridFTSSearcher requires SQLiteNeighborSearchStore',
      );
    }

    // Load the underlying RBPS searcher
    final rbpsSearcher = await RandomBinaryProjectionSearcher.loadFromStore(
      store,
      searcherId,
    );

    if (rbpsSearcher == null) {
      throw ArgumentError('Searcher with ID $searcherId not found');
    }

    return HybridFTSSearcher._(store, searcherId, rbpsSearcher);
  }

  /// Searches by keyword only (fast FTS search).
  ///
  /// Example:
  /// ```dart
  /// final results = await searcher.searchByKeyword('bonjour', k: 5);
  /// ```
  Future<List<TranslationResult>> searchByKeyword(
    String keyword, {
    int k = 10,
  }) async {
    if (keyword.trim().isEmpty) {
      throw ArgumentError('Keyword cannot be empty');
    }
    if (k <= 0) {
      throw ArgumentError('k must be greater than 0');
    }
    return await _keywordOnlySearch(keyword.trim(), k);
  }

  /// Searches by semantic similarity only (RBPS).
  ///
  /// Example:
  /// ```dart
  /// final queryEmbedding = Vector.fromList([0.1, 0.2, 0.3]);
  /// final results = await searcher.searchBySemantic(queryEmbedding, k: 5);
  /// ```
  Future<List<TranslationResult>> searchBySemantic(
    Vector embedding, {
    int k = 10,
    int searchRadius = 3,
    Distance distance = Distance.euclidean,
  }) async {
    if (k <= 0) {
      throw ArgumentError('k must be greater than 0');
    }
    if (searchRadius < 0) {
      throw ArgumentError('searchRadius must be non-negative');
    }
    return await _semanticOnlySearch(embedding, k, searchRadius, distance);
  }

  /// Performs hybrid search: keyword filtering + semantic ranking.
  ///
  /// This is the recommended method for best results - it filters by keywords
  /// first (fast), then ranks by semantic similarity.
  ///
  /// Example:
  /// ```dart
  /// final queryEmbedding = Vector.fromList([0.1, 0.2, 0.3]);
  /// final results = await searcher.searchHybrid(
  ///   keyword: 'bonjour',
  ///   embedding: queryEmbedding,
  ///   k: 10,
  /// );
  /// ```
  Future<List<TranslationResult>> searchHybrid({
    required String keyword,
    required Vector embedding,
    int k = 10,
    int searchRadius = 3,
    Distance distance = Distance.euclidean,
  }) async {
    if (keyword.trim().isEmpty) {
      throw ArgumentError('Keyword cannot be empty');
    }
    return await _hybridSearch(
        keyword.trim(), embedding, k, searchRadius, distance);
  }

  /// Gets the searcher ID.
  String get searcherId => _searcherId;

  /// Keyword-only search using FTS.
  Future<List<TranslationResult>> _keywordOnlySearch(
    String keywordQuery,
    int k,
  ) async {
    // Get FTS results
    final ftsIndices = _store.ftsSearch(_searcherId, keywordQuery);

    if (ftsIndices.isEmpty) {
      return [];
    }

    // Get text content for results
    final results = <TranslationResult>[];
    for (final index in ftsIndices.take(k)) {
      final textContent = await _store.getTextContent(_searcherId, index);
      if (textContent.sourceText != null && textContent.targetText != null) {
        results.add(TranslationResult(
          pointIndex: index,
          sourceText: textContent.sourceText!,
          targetText: textContent.targetText!,
          distance: 0.0, // No semantic distance for keyword-only
        ));
      }
    }

    return results;
  }

  /// Semantic-only search using RBPS.
  Future<List<TranslationResult>> _semanticOnlySearch(
    Vector semanticQuery,
    int k,
    int searchRadius,
    Distance distance,
  ) async {
    // Use RBPS to find nearest neighbors
    final neighbours = _rbpsSearcher.query(
      semanticQuery,
      k,
      searchRadius,
      distance: distance,
    );

    // Get text content for results
    final results = <TranslationResult>[];
    for (final neighbour in neighbours) {
      final textContent = await _store.getTextContent(
        _searcherId,
        neighbour.index,
      );
      if (textContent.sourceText != null && textContent.targetText != null) {
        results.add(TranslationResult(
          pointIndex: neighbour.index,
          sourceText: textContent.sourceText!,
          targetText: textContent.targetText!,
          distance: neighbour.distance.toDouble(),
        ));
      }
    }

    return results;
  }

  /// Hybrid search: FTS filtering + RBPS ranking.
  Future<List<TranslationResult>> _hybridSearch(
    String keywordQuery,
    Vector semanticQuery,
    int k,
    int searchRadius,
    Distance distance,
  ) async {
    // Step 1: FTS filtering
    final ftsIndices = _store.ftsSearch(_searcherId, keywordQuery);

    if (ftsIndices.isEmpty) {
      return [];
    }

    // Step 2: Use a larger k for RBPS to ensure we have enough candidates
    // after filtering by FTS results
    // Ensure we don't exceed available FTS results
    final maxK = ftsIndices.length;
    final expandedK = maxK < k ? maxK : (k * 3).clamp(k, maxK);

    // Step 3: Get semantic neighbors (already sorted by distance)
    final allNeighbours = _rbpsSearcher.query(
      semanticQuery,
      expandedK,
      searchRadius,
      distance: distance,
    );

    // Step 4: Filter to only FTS-matched indices and get text content
    final ftsSet = ftsIndices.toSet();
    final candidates = <TranslationResult>[];

    for (final neighbour in allNeighbours) {
      if (ftsSet.contains(neighbour.index)) {
        final textContent = await _store.getTextContent(
          _searcherId,
          neighbour.index,
        );
        if (textContent.sourceText != null && textContent.targetText != null) {
          candidates.add(TranslationResult(
            pointIndex: neighbour.index,
            sourceText: textContent.sourceText!,
            targetText: textContent.targetText!,
            distance: neighbour.distance.toDouble(),
          ));

          // Early exit when we have enough results
          if (candidates.length >= k) {
            break;
          }
        }
      }
    }

    // Results are already sorted by distance from RBPS query
    return candidates;
  }
}
