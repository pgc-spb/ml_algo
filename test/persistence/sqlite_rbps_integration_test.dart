import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('SQLite RandomBinaryProjectionSearcher Integration', () {
    late String testDbPath;
    late SQLiteNeighborSearchStore store;

    setUp(() {
      testDbPath =
          'test_integration_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);
    });

    tearDown(() async {
      store.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should perform full workflow: create → save → load → query',
        () async {
      // Create searcher
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ], headerExists: false);
      final originalSearcher = RandomBinaryProjectionSearcher(
        data,
        4,
        seed: 10,
      );

      // Query original searcher
      final queryPoint = Vector.fromList([23, 12, 34]);
      final k = 3;
      final searchRadius = 3;
      final originalNeighbours =
          originalSearcher.query(queryPoint, k, searchRadius);
      final originalList = originalNeighbours.toList();

      // Save using saveToStore method
      final searcherId = await originalSearcher.saveToStore(store);

      // Load using loadFromStore method
      final loadedSearcher =
          await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);

      expect(loadedSearcher, isNotNull);

      // Query loaded searcher
      final loadedNeighbours =
          loadedSearcher!.query(queryPoint, k, searchRadius);
      final loadedList = loadedNeighbours.toList();

      // Verify results match
      expect(loadedList.length, originalList.length);
      for (var i = 0; i < originalList.length; i++) {
        expect(loadedList[i].index, originalList[i].index);
        expect(
            loadedList[i].distance, closeTo(originalList[i].distance, 0.001));
      }
    });

    test('should handle multiple searchers in same database', () async {
      final data1 = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
      ], headerExists: false);
      final data2 = DataFrame([
        [10, 20, 30],
        [40, 50, 60],
      ], headerExists: false);

      final searcher1 = RandomBinaryProjectionSearcher(data1, 3, seed: 1);
      final searcher2 = RandomBinaryProjectionSearcher(data2, 3, seed: 2);

      final id1 = await searcher1.saveToStore(store, searcherId: 'searcher1');
      final id2 = await searcher2.saveToStore(store, searcherId: 'searcher2');

      expect(id1, 'searcher1');
      expect(id2, 'searcher2');

      final loaded1 = await RandomBinaryProjectionSearcher.loadFromStore(
          store, 'searcher1');
      final loaded2 = await RandomBinaryProjectionSearcher.loadFromStore(
          store, 'searcher2');

      expect(loaded1, isNotNull);
      expect(loaded2, isNotNull);
      expect(loaded1!.points.rowCount, 2);
      expect(loaded2!.points.rowCount, 2);

      // Verify they are different
      final queryPoint = Vector.fromList([1, 2, 3]);
      final neighbours1 = loaded1.query(queryPoint, 1, 2);
      final neighbours2 = loaded2.query(queryPoint, 1, 2);

      // Results should be different
      expect(neighbours1.first.index, isNot(equals(neighbours2.first.index)));
    });

    test('should preserve query results with different distance metrics',
        () async {
      final data = DataFrame([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);

      final searcherId = await searcher.saveToStore(store);
      final loadedSearcher =
          await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);

      expect(loadedSearcher, isNotNull);

      final queryPoint = Vector.fromList([2.0, 3.0, 4.0]);

      // Test euclidean distance
      final euclideanNeighbours =
          loadedSearcher!.query(queryPoint, 2, 2, distance: Distance.euclidean);
      expect(euclideanNeighbours.length, 2);

      // Test cosine distance
      final cosineNeighbours =
          loadedSearcher.query(queryPoint, 2, 2, distance: Distance.cosine);
      expect(cosineNeighbours.length, 2);

      // Results should be different for different metrics
      expect(euclideanNeighbours.first.index,
          isNot(equals(cosineNeighbours.first.index)));
    });

    test('should handle searcher updates (save with same ID)', () async {
      final data1 = DataFrame([
        [1, 2],
        [3, 4],
      ], headerExists: false);
      final data2 = DataFrame([
        [10, 20],
        [30, 40],
      ], headerExists: false);

      final searcher1 = RandomBinaryProjectionSearcher(data1, 3);
      final searcher2 = RandomBinaryProjectionSearcher(data2, 3);

      final id = 'same-id';
      await searcher1.saveToStore(store, searcherId: id);
      await searcher2.saveToStore(store, searcherId: id);

      final loaded =
          await RandomBinaryProjectionSearcher.loadFromStore(store, id);

      expect(loaded, isNotNull);
      expect(loaded!.points.rowCount, 2);
      // Should have data2 values
      expect(loaded.points[0][0], closeTo(10, 0.001));
    });
  });
}
