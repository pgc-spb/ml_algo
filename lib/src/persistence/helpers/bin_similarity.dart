/// Helper functions for calculating bin similarity using Hamming distance.
///
/// Bin similarity is used as an approximate distance metric to filter
/// candidates before loading actual points from SQLite.

/// Calculates the Hamming distance between two bin IDs.
///
/// The Hamming distance is the number of positions at which the corresponding
/// bits are different. This provides an approximate measure of how similar
/// two bins are, which correlates with point similarity.
///
/// Returns a non-negative integer representing the Hamming distance.
/// Lower values indicate more similar bins.
int hammingDistance(int binId1, int binId2) {
  // XOR to find differing bits, then count set bits
  final xor = binId1 ^ binId2;
  return xor.toRadixString(2).split('').where((bit) => bit == '1').length;
}

/// Calculates bin similarity score (inverse of Hamming distance).
///
/// Returns a score where higher values indicate more similar bins.
/// This is useful for sorting candidates by approximate similarity.
double binSimilarity(int binId1, int binId2) {
  final distance = hammingDistance(binId1, binId2);
  // Convert to similarity: 1 / (1 + distance)
  // This gives a score between 0 and 1, where 1 is identical
  return 1.0 / (1.0 + distance);
}

/// Sorts candidate indices by bin similarity to a query bin.
///
/// [candidates] is a list of (pointIndex, binId) pairs.
/// [queryBinId] is the bin ID of the query point.
///
/// Returns the candidates sorted by similarity (most similar first).
List<int> sortCandidatesByBinSimilarity(
  List<MapEntry<int, int>> candidates,
  int queryBinId,
) {
  // Sort by Hamming distance (lower is better)
  candidates.sort((a, b) {
    final distA = hammingDistance(queryBinId, a.value);
    final distB = hammingDistance(queryBinId, b.value);
    return distA.compareTo(distB);
  });
  
  // Return just the point indices
  return candidates.map((entry) => entry.key).toList();
}

