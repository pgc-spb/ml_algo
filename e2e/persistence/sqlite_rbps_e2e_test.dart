import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('SQLite RandomBinaryProjectionSearcher E2E', () {
    late String testDbPath;
    late SQLiteNeighborSearchStore store;

    setUp(() {
      testDbPath = 'test_e2e_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);
    });

    tearDown(() async {
      store.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should handle large dataset (1000+ vectors)', () async {
      // Create a large dataset
      final rows = <List<num>>[];
      for (var i = 0; i < 1000; i++) {
        rows.add([i * 1.0, i * 2.0, i * 3.0, i * 4.0]);
      }
      final data = DataFrame(rows, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 6, seed: 42);

      // Save
      final searcherId = await searcher.saveToStore(store);

      // Load
      final loadedSearcher =
          await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);

      expect(loadedSearcher, isNotNull);
      expect(loadedSearcher!.points.rowCount, 1000);
      expect(loadedSearcher.points.columnCount, 4);

      // Query
      final queryPoint = Vector.fromList([500.0, 1000.0, 1500.0, 2000.0]);
      final neighbours = loadedSearcher.query(queryPoint, 10, 3);

      expect(neighbours.length, 10);
      // The nearest neighbour should be around index 500
      final nearestIndex = neighbours.first.index;
      expect(nearestIndex, greaterThanOrEqualTo(490));
      expect(nearestIndex, lessThanOrEqualTo(510));
    });

    test('should handle phrase translation-like scenario', () async {
      // Simulate phrase translation: each row is a phrase embedding (e.g., 768 dimensions)
      // For testing, we'll use smaller dimensions
      final dimension = 10;
      final phraseCount = 100;

      final rows = <List<num>>[];
      for (var i = 0; i < phraseCount; i++) {
        final phrase = <num>[];
        for (var j = 0; j < dimension; j++) {
          phrase.add(i * dimension + j);
        }
        rows.add(phrase);
      }

      final data = DataFrame(rows, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 8, seed: 123);

      // Save
      final searcherId = await searcher.saveToStore(store);

      // Load
      final loadedSearcher =
          await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);

      expect(loadedSearcher, isNotNull);

      // Query for similar phrases (simulate translation lookup)
      final queryPhrase = Vector.fromList(
          [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0]);
      final similarPhrases = loadedSearcher!.query(queryPhrase, 5, 3);

      expect(similarPhrases.length, 5);
      // Should find phrases around index 5 (since 50/10 = 5)
      final nearestIndex = similarPhrases.first.index;
      expect(nearestIndex, greaterThanOrEqualTo(3));
      expect(nearestIndex, lessThanOrEqualTo(7));
    });

    test('should maintain performance with multiple queries', () async {
      final rows = <List<num>>[];
      for (var i = 0; i < 500; i++) {
        rows.add([i * 1.0, i * 2.0, i * 3.0]);
      }
      final data = DataFrame(rows, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 6);

      final searcherId = await searcher.saveToStore(store);
      final loadedSearcher =
          await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);

      expect(loadedSearcher, isNotNull);

      // Perform multiple queries
      for (var i = 0; i < 10; i++) {
        final queryPoint = Vector.fromList([i * 50.0, i * 100.0, i * 150.0]);
        final neighbours = loadedSearcher!.query(queryPoint, 5, 3);
        expect(neighbours.length, 5);
      }
    });

    test('should handle float64 dtype in large dataset', () async {
      final rows = <List<num>>[];
      for (var i = 0; i < 500; i++) {
        rows.add([i * 1.0, i * 2.0, i * 3.0]);
      }
      final data = DataFrame(rows, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(
        data,
        6,
        dtype: DType.float64,
        seed: 999,
      );

      final searcherId = await searcher.saveToStore(store);
      final loadedSearcher =
          await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);

      expect(loadedSearcher, isNotNull);
      expect(loadedSearcher!.points.dtype, DType.float64);

      final queryPoint = Vector.fromList([250.0, 500.0, 750.0]);
      final neighbours = loadedSearcher.query(queryPoint, 5, 3);

      expect(neighbours.length, 5);
    });

    test('should handle metadata operations', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 42);

      final searcherId = await searcher.saveToStore(store);

      // Get metadata
      final metadata = await store.getSearcherMetadata(searcherId);
      expect(metadata, isNotNull);
      expect(metadata!['digitCapacity'], 4);
      expect(metadata['seed'], 42);
      expect(metadata['pointCount'], 2);
      expect(metadata['columnCount'], 3);
      expect(metadata['columns'], isA<List<String>>());

      // List searchers
      final searchers = await store.listSearchers();
      expect(searchers, contains(searcherId));

      // Delete and verify
      final deleted = await store.deleteSearcher(searcherId);
      expect(deleted, isTrue);

      final searchersAfterDelete = await store.listSearchers();
      expect(searchersAfterDelete, isNot(contains(searcherId)));

      final metadataAfterDelete = await store.getSearcherMetadata(searcherId);
      expect(metadataAfterDelete, isNull);
    });
  });
}
