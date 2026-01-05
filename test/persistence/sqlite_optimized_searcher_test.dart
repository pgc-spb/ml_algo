import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('SQLiteOptimizedSearcher', () {
    late String testDbPath;
    late SQLiteNeighborSearchStore store;

    setUp(() {
      testDbPath = 'test_optimized_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);
    });

    tearDown(() async {
      store.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should load optimized searcher', () async {
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 10);
      final searcherId = await searcher.saveToStore(store);

      final optimized = await store.loadOptimizedSearcher(searcherId);

      expect(optimized, isNotNull);
      expect(optimized!.digitCapacity, 4);
      expect(optimized.seed, 10);
      expect(optimized.columns, data.header);
    });

    test('should return null for non-existent searcher', () async {
      final optimized = await store.loadOptimizedSearcher('non-existent');
      expect(optimized, isNull);
    });

    test('should perform queries correctly', () async {
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 10);
      final searcherId = await searcher.saveToStore(store);

      final optimized = await store.loadOptimizedSearcher(searcherId);
      final queryPoint = Vector.fromList([23, 12, 34]);
      final k = 3;
      final searchRadius = 3;

      final neighbours = optimized!.query(queryPoint, k, searchRadius);

      expect(neighbours, isNotEmpty);
      expect(neighbours.length, lessThanOrEqualTo(k));

      // First neighbour should be the query point itself (distance 0)
      final first = neighbours.first;
      expect(first.index, 0);
      expect(first.distance, closeTo(0.0, 0.001));
    });

    test('should match results with regular searcher', () async {
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ], headerExists: false);
      final originalSearcher =
          RandomBinaryProjectionSearcher(data, 4, seed: 10);
      final queryPoint = Vector.fromList([23, 12, 34]);
      final k = 3;
      final searchRadius = 3;

      // Query original searcher
      final originalNeighbours =
          originalSearcher.query(queryPoint, k, searchRadius);
      final originalList = originalNeighbours.toList();

      // Save and load optimized
      final searcherId = await originalSearcher.saveToStore(store);
      final optimized = await store.loadOptimizedSearcher(searcherId);

      // Query optimized searcher
      final optimizedNeighbours = optimized!.query(queryPoint, k, searchRadius);
      final optimizedList = optimizedNeighbours.toList();

      // Results should match (same indices and similar distances)
      expect(optimizedList.length, originalList.length);
      for (var i = 0; i < originalList.length; i++) {
        expect(optimizedList[i].index, originalList[i].index);
        expect(
            optimizedList[i].distance, closeTo(originalList[i].distance, 0.1));
      }
    });

    test('should handle different k values', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4);
      final searcherId = await searcher.saveToStore(store);
      final optimized = await store.loadOptimizedSearcher(searcherId);
      final queryPoint = Vector.fromList([2, 3, 4]);

      final k1 = optimized!.query(queryPoint, 1, 3);
      expect(k1.length, 1);

      final k3 = optimized.query(queryPoint, 3, 3);
      expect(k3.length, 3);

      final k5 = optimized.query(queryPoint, 5, 3);
      expect(k5.length, lessThanOrEqualTo(5));
    });

    test('should handle different search radius', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4);
      final searcherId = await searcher.saveToStore(store);
      final optimized = await store.loadOptimizedSearcher(searcherId);
      final queryPoint = Vector.fromList([2, 3, 4]);

      final radius0 = optimized!.query(queryPoint, 3, 0);
      final radius1 = optimized.query(queryPoint, 3, 1);
      final radius3 = optimized.query(queryPoint, 3, 3);

      // Larger radius should find more candidates (or same)
      expect(radius3.length, greaterThanOrEqualTo(radius1.length));
      expect(radius1.length, greaterThanOrEqualTo(radius0.length));
    });

    test('should handle different distance metrics', () async {
      final data = DataFrame([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);
      final searcherId = await searcher.saveToStore(store);
      final optimized = await store.loadOptimizedSearcher(searcherId);
      final queryPoint = Vector.fromList([2.0, 3.0, 4.0]);

      final euclidean =
          optimized!.query(queryPoint, 2, 2, distance: Distance.euclidean);
      final cosine =
          optimized.query(queryPoint, 2, 2, distance: Distance.cosine);

      expect(euclidean.length, 2);
      expect(cosine.length, 2);

      // Different metrics may give different results
      // Just verify they both work
      expect(euclidean.first.distance, greaterThanOrEqualTo(0));
      expect(cosine.first.distance, greaterThanOrEqualTo(0));
    });

    test('should throw error when accessing points property', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);
      final searcherId = await searcher.saveToStore(store);
      final optimized = await store.loadOptimizedSearcher(searcherId);

      expect(() => optimized!.points, throwsA(isA<UnsupportedError>()));
    });

    test('should handle large dataset', () async {
      // Create a larger dataset
      final rows = <List<num>>[];
      for (var i = 0; i < 500; i++) {
        rows.add([i * 1.0, i * 2.0, i * 3.0]);
      }
      final data = DataFrame(rows, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 6);
      final searcherId = await searcher.saveToStore(store);

      final optimized = await store.loadOptimizedSearcher(searcherId);
      expect(optimized, isNotNull);

      // Query should work
      final queryPoint = Vector.fromList([250.0, 500.0, 750.0]);
      final neighbours = optimized!.query(queryPoint, 5, 3);
      expect(neighbours.length, 5);
    });

    test('should use custom approximateFilterLimit', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);
      final searcherId = await searcher.saveToStore(store);

      final optimized = await store.loadOptimizedSearcher(
        searcherId,
        approximateFilterLimit: 100,
        cacheSize: 50,
      );

      expect(optimized, isNotNull);
      final queryPoint = Vector.fromList([2, 3, 4]);
      final neighbours = optimized!.query(queryPoint, 2, 2);
      expect(neighbours.length, 2);
    });

    test('should handle float64 dtype', () async {
      final data = DataFrame([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(
        data,
        3,
        dtype: DType.float64,
      );
      final searcherId = await searcher.saveToStore(store);

      final optimized = await store.loadOptimizedSearcher(searcherId);
      expect(optimized, isNotNull);

      final queryPoint = Vector.fromList([2.0, 3.0, 4.0]);
      final neighbours = optimized!.query(queryPoint, 2, 2);
      expect(neighbours.length, 2);
    });
  });
}
