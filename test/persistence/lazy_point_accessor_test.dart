import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('SQLiteLazyPointAccessor', () {
    late String testDbPath;
    late SQLiteNeighborSearchStore store;
    late String searcherId;

    setUp(() async {
      testDbPath = 'test_lazy_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);

      // Create and save a searcher for testing
      final data = DataFrame([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 10);
      searcherId = await searcher.saveToStore(store);
    });

    tearDown(() async {
      store.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should work through optimized searcher', () async {
      // Test lazy point accessor through optimized searcher
      final optimized = await store.loadOptimizedSearcher(searcherId);
      expect(optimized, isNotNull);
      
      // Test that we can query (which uses the lazy accessor internally)
      final queryPoint = Vector.fromList([1.0, 2.0, 3.0]);
      final neighbours = optimized!.query(queryPoint, 2, 2);
      expect(neighbours.length, 2);
      
      // Verify results are correct
      final first = neighbours.first;
      expect(first.index, 0); // Should match the query point
      expect(first.distance, closeTo(0.0, 0.001));
    });
  });
}
