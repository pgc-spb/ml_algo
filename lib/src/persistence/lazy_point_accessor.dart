import 'dart:typed_data';
import 'package:ml_algo/src/persistence/helpers/matrix_serialization.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:sqlite3/sqlite3.dart';

/// Interface for lazy loading of points from SQLite.
///
/// This interface allows loading points on-demand during queries,
/// avoiding the need to load all points into memory at once.
abstract class LazyPointAccessor {
  /// Gets a single point by index.
  ///
  /// May load from SQLite if not cached.
  Future<Vector> getPoint(int index);

  /// Gets multiple points by indices.
  ///
  /// This method should batch load points for efficiency.
  /// May load from SQLite if not cached.
  Future<List<Vector>> getPoints(List<int> indices);

  /// Gets a cached point if available, or null if not cached.
  ///
  /// This allows synchronous access to frequently-used points
  /// without triggering SQLite I/O.
  Vector? getCached(int index);

  /// Gets the total number of points.
  int get pointCount;

  /// Gets the data type of the points.
  DType get dtype;

  /// Gets the number of dimensions (columns) in each point.
  int get columnCount;
}

/// SQLite-based implementation of [LazyPointAccessor].
///
/// This implementation loads points from SQLite on-demand,
/// with optional LRU caching for frequently accessed points.
class SQLiteLazyPointAccessor implements LazyPointAccessor {
  final Database _db;
  final String _searcherId;
  final DType _dtype;
  final int _columnCount;
  final int _pointCount;
  
  // LRU cache: Map<pointIndex, Vector>
  final Map<int, Vector> _cache;
  final int _maxCacheSize;
  final List<int> _cacheAccessOrder; // For LRU eviction

  SQLiteLazyPointAccessor(
    this._db,
    this._searcherId,
    this._dtype,
    this._columnCount,
    this._pointCount, {
    int maxCacheSize = 1000,
  })  : _cache = {},
        _maxCacheSize = maxCacheSize,
        _cacheAccessOrder = [];

  @override
  int get pointCount => _pointCount;

  @override
  DType get dtype => _dtype;

  @override
  int get columnCount => _columnCount;

  @override
  Vector? getCached(int index) {
    if (_cache.containsKey(index)) {
      // Update access order for LRU
      _cacheAccessOrder.remove(index);
      _cacheAccessOrder.add(index);
      return _cache[index];
    }
    return null;
  }

  @override
  Future<Vector> getPoint(int index) async {
    // Check cache first
    final cached = getCached(index);
    if (cached != null) {
      return cached;
    }

    // Load from SQLite
    final stmt = _db.prepare('''
      SELECT vector_data
      FROM searcher_points
      WHERE searcher_id = ? AND point_index = ?
    ''');
    
    try {
      final rows = stmt.select([_searcherId, index]);
      if (rows.isEmpty) {
        throw RangeError('Point index $index out of range');
      }
      
      final blob = rows.first[0] as Uint8List;
      final rowValues = deserializeMatrixRow(blob);
      final vector = Vector.fromList(rowValues, dtype: _dtype);
      
      // Cache the point
      _cachePoint(index, vector);
      
      return vector;
    } finally {
      stmt.dispose();
    }
  }

  @override
  Future<List<Vector>> getPoints(List<int> indices) async {
    if (indices.isEmpty) {
      return [];
    }

    // Separate cached and uncached indices
    final uncachedIndices = <int>[];
    final indexToResult = <int, Vector>{};

    for (final index in indices) {
      final cached = getCached(index);
      if (cached != null) {
        indexToResult[index] = cached;
      } else {
        uncachedIndices.add(index);
      }
    }

    // Batch load uncached points
    if (uncachedIndices.isNotEmpty) {
      // Build IN clause with placeholders
      final placeholders = uncachedIndices.map((_) => '?').join(', ');
      final stmt = _db.prepare('''
        SELECT point_index, vector_data
        FROM searcher_points
        WHERE searcher_id = ? AND point_index IN ($placeholders)
        ORDER BY point_index
      ''');

      try {
        final args = [_searcherId, ...uncachedIndices];
        final rows = stmt.select(args);

        for (final row in rows) {
          final pointIndex = row[0] as int;
          final blob = row[1] as Uint8List;
          final rowValues = deserializeMatrixRow(blob);
          final vector = Vector.fromList(rowValues, dtype: _dtype);
          
          indexToResult[pointIndex] = vector;
          _cachePoint(pointIndex, vector);
        }
      } finally {
        stmt.dispose();
      }
    }

    // Return vectors in the same order as requested indices
    return indices.map((index) => indexToResult[index]!).toList();
  }

  /// Caches a point with LRU eviction.
  void _cachePoint(int index, Vector vector) {
    // Remove if already exists (update)
    if (_cache.containsKey(index)) {
      _cacheAccessOrder.remove(index);
    } else if (_cache.length >= _maxCacheSize) {
      // Evict least recently used
      final lruIndex = _cacheAccessOrder.removeAt(0);
      _cache.remove(lruIndex);
    }

    _cache[index] = vector;
    _cacheAccessOrder.add(index);
  }

  /// Clears the cache.
  void clearCache() {
    _cache.clear();
    _cacheAccessOrder.clear();
  }
}

