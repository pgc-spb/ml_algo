import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/neighbor_search_store.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('SQLiteNeighborSearchStore', () {
    late String testDbPath;
    late NeighborSearchStore store;

    setUp(() {
      // Create a temporary database file for each test
      testDbPath = 'test_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);
    });

    tearDown(() async {
      // Clean up test database
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should save and load a searcher', () async {
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ], headerExists: false);
      final digitCapacity = 4;
      final searcher = RandomBinaryProjectionSearcher(
        data,
        digitCapacity,
        seed: 10,
      );

      // Save searcher
      final searcherId = await store.saveSearcher(searcher);

      expect(searcherId, isNotEmpty);

      // Load searcher
      final loadedSearcher = await store.loadSearcher(searcherId);

      expect(loadedSearcher, isNotNull);
      expect(loadedSearcher!.digitCapacity, digitCapacity);
      expect(loadedSearcher.seed, 10);
      expect(loadedSearcher.columns, data.header);
      expect(loadedSearcher.points.rowCount, data.rows.length);
      expect(loadedSearcher.points.columnCount, data.header.length);
    });

    test('should save searcher with custom ID', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);
      final customId = 'my-custom-id';

      final searcherId =
          await store.saveSearcher(searcher, searcherId: customId);

      expect(searcherId, customId);

      final loadedSearcher = await store.loadSearcher(customId);
      expect(loadedSearcher, isNotNull);
    });

    test('should return null when loading non-existent searcher', () async {
      final loadedSearcher = await store.loadSearcher('non-existent-id');
      expect(loadedSearcher, isNull);
    });

    test('should delete a searcher', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);

      final searcherId = await store.saveSearcher(searcher);
      expect(await store.loadSearcher(searcherId), isNotNull);

      final deleted = await store.deleteSearcher(searcherId);
      expect(deleted, isTrue);

      expect(await store.loadSearcher(searcherId), isNull);
    });

    test('should return false when deleting non-existent searcher', () async {
      final deleted = await store.deleteSearcher('non-existent-id');
      expect(deleted, isFalse);
    });

    test('should list all searchers', () async {
      final data1 = DataFrame([
        [1, 2],
        [3, 4]
      ], headerExists: false);
      final data2 = DataFrame([
        [5, 6],
        [7, 8]
      ], headerExists: false);
      final searcher1 = RandomBinaryProjectionSearcher(data1, 3);
      final searcher2 = RandomBinaryProjectionSearcher(data2, 3);

      final id1 = await store.saveSearcher(searcher1, searcherId: 'id1');
      final id2 = await store.saveSearcher(searcher2, searcherId: 'id2');

      final searchers = await store.listSearchers();
      expect(searchers, containsAll([id1, id2]));
      expect(searchers.length, 2);
    });

    test('should return empty list when no searchers exist', () async {
      final searchers = await store.listSearchers();
      expect(searchers, isEmpty);
    });

    test('should get searcher metadata', () async {
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
      ], headerExists: false);
      final digitCapacity = 4;
      final seed = 10;
      final searcher = RandomBinaryProjectionSearcher(
        data,
        digitCapacity,
        seed: seed,
      );

      final searcherId = await store.saveSearcher(searcher);
      final metadata = await store.getSearcherMetadata(searcherId);

      expect(metadata, isNotNull);
      expect(metadata!['digitCapacity'], digitCapacity);
      expect(metadata['seed'], seed);
      expect(metadata['columnCount'], 3);
      expect(metadata['pointCount'], 3);
      expect(metadata['dtype'], 'float32');
      expect(metadata['columns'], isA<List<String>>());
      expect(metadata['createdAt'], isA<String>());
    });

    test('should return null when getting metadata for non-existent searcher',
        () async {
      final metadata = await store.getSearcherMetadata('non-existent-id');
      expect(metadata, isNull);
    });

    test('should preserve query results after save and load', () async {
      final data = DataFrame([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 10);
      final queryPoint = Vector.fromList([23, 12, 34]);
      final k = 3;
      final searchRadius = 3;

      // Query original searcher
      final originalNeighbours = searcher.query(queryPoint, k, searchRadius);

      // Save and load
      final searcherId = await store.saveSearcher(searcher);
      final loadedSearcher = await store.loadSearcher(searcherId);

      // Query loaded searcher
      final loadedNeighbours =
          loadedSearcher!.query(queryPoint, k, searchRadius);

      // Results should match
      expect(loadedNeighbours.length, originalNeighbours.length);
      final originalList = originalNeighbours.toList();
      final loadedList = loadedNeighbours.toList();

      for (var i = 0; i < originalList.length; i++) {
        expect(loadedList[i].index, originalList[i].index);
        expect(
            loadedList[i].distance, closeTo(originalList[i].distance, 0.001));
      }
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

      final searcherId = await store.saveSearcher(searcher);
      final loadedSearcher = await store.loadSearcher(searcherId);

      expect(loadedSearcher, isNotNull);
      expect(loadedSearcher!.points.dtype, DType.float64);

      final metadata = await store.getSearcherMetadata(searcherId);
      expect(metadata!['dtype'], 'float64');
    });

    test('should handle searcher without seed', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 3);

      final searcherId = await store.saveSearcher(searcher);
      final loadedSearcher = await store.loadSearcher(searcherId);

      expect(loadedSearcher, isNotNull);
      expect(loadedSearcher!.seed, isNull);

      final metadata = await store.getSearcherMetadata(searcherId);
      expect(metadata!['seed'], isNull);
    });

    test('should handle large dataset', () async {
      // Create a larger dataset
      final rows = <List<num>>[];
      for (var i = 0; i < 1000; i++) {
        rows.add([i * 1.0, i * 2.0, i * 3.0]);
      }
      final data = DataFrame(rows, headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 6);

      final searcherId = await store.saveSearcher(searcher);
      final loadedSearcher = await store.loadSearcher(searcherId);

      expect(loadedSearcher, isNotNull);
      expect(loadedSearcher!.points.rowCount, 1000);
      expect(loadedSearcher.points.columnCount, 3);

      // Verify query still works
      final queryPoint = Vector.fromList([500.0, 1000.0, 1500.0]);
      final neighbours = loadedSearcher.query(queryPoint, 5, 3);
      expect(neighbours.length, 5);
    });
  });
}
