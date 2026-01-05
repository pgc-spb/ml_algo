import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_algo/src/retrieval/hybrid_fts_searcher.dart';
import 'package:ml_algo/src/retrieval/translation_pair.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('HybridFTSSearcher', () {
    late String testDbPath;
    late SQLiteNeighborSearchStore store;
    late String searcherId;

    setUp(() async {
      testDbPath =
          'test_hybrid_fts_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);

      // Create test data: French-English translation pairs with text content
      final translations = [
        [
          'Bonjour',
          'Hello',
          [0.1, 0.2, 0.3]
        ],
        [
          'Comment allez-vous?',
          'How are you?',
          [0.2, 0.3, 0.4]
        ],
        [
          'Je vais bien',
          'I am fine',
          [0.3, 0.4, 0.5]
        ],
        [
          'Merci beaucoup',
          'Thank you very much',
          [0.4, 0.5, 0.6]
        ],
        [
          'Au revoir',
          'Goodbye',
          [0.5, 0.6, 0.7]
        ],
        [
          'Bonjour le monde',
          'Hello world',
          [0.15, 0.25, 0.35]
        ],
        [
          'Le chat',
          'The cat',
          [0.6, 0.7, 0.8]
        ],
        [
          'Le chien',
          'The dog',
          [0.7, 0.8, 0.9]
        ],
      ];

      // Create DataFrame with embeddings
      final embeddings = translations.map((t) => t[2] as List<double>).toList();
      final data = DataFrame(embeddings, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 42);

      // Save searcher
      searcherId = await searcher.saveToStore(store);

      // Add text content for FTS
      for (var i = 0; i < translations.length; i++) {
        await store.addTextContent(
          searcherId,
          i,
          frenchText: translations[i][0] as String,
          englishText: translations[i][1] as String,
        );
      }
    });

    tearDown(() async {
      store.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should create HybridFTSSearcher from store', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);
      expect(hybridSearcher, isNotNull);
    });

    test('should perform FTS keyword search only', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);

      final results = await hybridSearcher.searchByKeyword('Bonjour', k: 5);

      expect(results, isNotEmpty);
      expect(results.length, lessThanOrEqualTo(5));
      // Should find phrases containing "Bonjour"
      expect(results.any((r) => r.frenchText.contains('Bonjour')), isTrue);
    });

    test('should perform semantic search only', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);

      final queryVector = Vector.fromList([0.1, 0.2, 0.3]);
      final results = await hybridSearcher.searchBySemantic(
        queryVector,
        k: 3,
        searchRadius: 3,
      );

      expect(results, isNotEmpty);
      expect(results.length, lessThanOrEqualTo(3));
      // First result should be most similar
      expect(results.first.distance, greaterThanOrEqualTo(0));
    });

    test('should perform hybrid FTS + semantic search', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);

      final queryVector = Vector.fromList([0.15, 0.25, 0.35]);
      final results = await hybridSearcher.searchHybrid(
        keyword: 'Bonjour',
        embedding: queryVector,
        k: 3,
        searchRadius: 3,
      );

      expect(results, isNotEmpty);
      expect(results.length, lessThanOrEqualTo(3));
      // All results should contain keyword
      for (final result in results) {
        expect(result.frenchText.toLowerCase(), contains('bonjour'));
      }
      // Results should be ranked by semantic similarity
      for (var i = 0; i < results.length - 1; i++) {
        expect(results[i].distance, lessThanOrEqualTo(results[i + 1].distance));
      }
    });

    test('should handle semantic-only query', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);

      final queryVector = Vector.fromList([0.1, 0.2, 0.3]);
      final results = await hybridSearcher.searchBySemantic(queryVector, k: 5);

      expect(results, isNotEmpty);
    });

    test('should handle empty semantic query', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);

      final results = await hybridSearcher.searchByKeyword('chat', k: 5);

      expect(results, isNotEmpty);
      expect(results.any((r) => r.frenchText.contains('chat')), isTrue);
    });

    test('should return TranslationResult with all fields', () async {
      final hybridSearcher =
          await HybridFTSSearcher.loadFromStore(store, searcherId);

      final results = await hybridSearcher.searchByKeyword('Bonjour', k: 1);

      expect(results, isNotEmpty);
      final result = results.first;
      expect(result.pointIndex, isNotNull);
      expect(result.frenchText, isNotEmpty);
      expect(result.englishText, isNotEmpty);
      expect(result.distance, greaterThanOrEqualTo(0));
    });

    test('should create searcher using simplified createFromTranslations',
        () async {
      final simpleStore = SQLiteNeighborSearchStore(
          'test_simple_${DateTime.now().millisecondsSinceEpoch}.db');

      final translations = [
        TranslationPair(
            french: 'Bonjour', english: 'Hello', embedding: [0.1, 0.2, 0.3]),
        TranslationPair(
            french: 'Au revoir',
            english: 'Goodbye',
            embedding: [0.4, 0.5, 0.6]),
        TranslationPair(
            french: 'Merci', english: 'Thank you', embedding: [0.7, 0.8, 0.9]),
      ];

      final searcher = await HybridFTSSearcher.createFromTranslations(
        simpleStore,
        translations,
        digitCapacity: 4,
        searcherId: 'simple-test',
      );

      // Test simplified search methods
      final keywordResults = await searcher.searchByKeyword('Bonjour');
      expect(keywordResults, isNotEmpty);
      expect(keywordResults.first.frenchText, contains('Bonjour'));

      final semanticResults = await searcher.searchBySemantic(
        Vector.fromList([0.1, 0.2, 0.3]),
        k: 2,
      );
      expect(semanticResults, isNotEmpty);
      expect(semanticResults.length, lessThanOrEqualTo(2));

      final hybridResults = await searcher.searchHybrid(
        keyword: 'Merci',
        embedding: Vector.fromList([0.7, 0.8, 0.9]),
        k: 1,
      );
      expect(hybridResults, isNotEmpty);
      expect(hybridResults.first.frenchText, contains('Merci'));

      simpleStore.close();
      // Note: We can't access _dbPath directly, but the test DB will be cleaned up
      // by the test framework or manually if needed
    });
  });
}
