import 'package:ml_algo/src/persistence/helpers/bin_similarity.dart';
import 'package:test/test.dart';

void main() {
  group('Bin Similarity', () {
    test('hammingDistance should calculate correct distance', () {
      // Same bin IDs
      expect(hammingDistance(0, 0), 0);
      expect(hammingDistance(5, 5), 0);
      expect(hammingDistance(255, 255), 0);

      // Different bin IDs
      expect(hammingDistance(0, 1), 1); // 0000 vs 0001
      expect(hammingDistance(1, 2), 2); // 0001 vs 0010
      expect(hammingDistance(3, 12), 4); // 0011 vs 1100 (all bits differ)
      expect(hammingDistance(15, 0), 4); // 1111 vs 0000
    });

    test('binSimilarity should return higher values for similar bins', () {
      // Identical bins
      expect(binSimilarity(0, 0), 1.0);
      expect(binSimilarity(5, 5), 1.0);

      // Very similar bins (distance 1)
      final sim1 = binSimilarity(0, 1);
      expect(sim1, greaterThan(0.0));
      expect(sim1, lessThan(1.0));

      // Less similar bins (distance 2)
      final sim2 = binSimilarity(0, 3);
      expect(sim2, lessThan(sim1));

      // Very different bins (distance 4)
      final sim4 = binSimilarity(0, 15);
      expect(sim4, lessThan(sim2));
    });

    test('sortCandidatesByBinSimilarity should sort correctly', () {
      final queryBinId = 0; // 0000
      final candidates = [
        MapEntry(0, 15), // 1111 - distance 4
        MapEntry(1, 0),  // 0000 - distance 0 (best)
        MapEntry(2, 3),  // 0011 - distance 2
        MapEntry(3, 1),  // 0001 - distance 1
        MapEntry(4, 7),  // 0111 - distance 3
      ];

      final sorted = sortCandidatesByBinSimilarity(candidates, queryBinId);

      // Should be sorted by Hamming distance (lower is better)
      expect(sorted[0], 1); // distance 0
      expect(sorted[1], 3); // distance 1
      expect(sorted[2], 2); // distance 2
      expect(sorted[3], 4); // distance 3
      expect(sorted[4], 0); // distance 4
    });

    test('sortCandidatesByBinSimilarity should handle empty list', () {
      final sorted = sortCandidatesByBinSimilarity([], 0);
      expect(sorted, isEmpty);
    });

    test('sortCandidatesByBinSimilarity should handle single candidate', () {
      final candidates = [MapEntry(0, 5)];
      final sorted = sortCandidatesByBinSimilarity(candidates, 5);
      expect(sorted, [0]);
    });
  });
}

