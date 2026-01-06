import 'dart:typed_data';
import 'package:ml_algo/src/persistence/helpers/bin_map_serialization.dart';
import 'package:ml_algo/src/persistence/helpers/dtype_converter.dart';
import 'package:ml_algo/src/persistence/helpers/matrix_serialization.dart';
import 'package:ml_algo/src/persistence/lazy_point_accessor.dart';
import 'package:ml_algo/src/persistence/neighbor_search_store.dart';
import 'package:ml_algo/src/persistence/sqlite_optimized_searcher.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_binary_representation.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_indices_from_binary_representation.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:sqlite3/sqlite3.dart';

/// Text content for a translation point.
///
/// Generic naming: source and target can be any languages.
class TextContent {
  final String? sourceText;
  final String? targetText;

  TextContent(this.sourceText, this.targetText);
}

/// SQLite implementation of [NeighborSearchStore].
///
/// This implementation stores [RandomBinaryProjectionSearcher] instances in a SQLite database,
/// enabling efficient storage and retrieval for large-scale applications.
///
/// Example:
///
/// ```dart
/// final store = SQLiteNeighborSearchStore('path/to/database.db');
/// final searcherId = await searcher.saveToStore(store);
/// final loadedSearcher = await RandomBinaryProjectionSearcher.loadFromStore(store, searcherId);
/// ```
class SQLiteNeighborSearchStore implements NeighborSearchStore {
  final String _dbPath;
  Database? _db;

  /// Creates a new [SQLiteNeighborSearchStore] instance.
  ///
  /// [dbPath] is the path to the SQLite database file.
  /// If the database doesn't exist, it will be created.
  SQLiteNeighborSearchStore(this._dbPath) {
    _initializeDatabase();
  }

  void _initializeDatabase() {
    _db = sqlite3.open(_dbPath);
    _createTables();
  }

  void _createTables() {
    final db = _db!;

    // Main searcher metadata table
    db.execute('''
      CREATE TABLE IF NOT EXISTS neighbor_searchers (
        id TEXT PRIMARY KEY,
        digit_capacity INTEGER NOT NULL,
        seed INTEGER,
        schema_version INTEGER NOT NULL,
        dtype TEXT NOT NULL,
        column_count INTEGER NOT NULL,
        point_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        metadata TEXT
      )
    ''');

    // Store column names
    db.execute('''
      CREATE TABLE IF NOT EXISTS searcher_columns (
        searcher_id TEXT NOT NULL,
        column_index INTEGER NOT NULL,
        column_name TEXT NOT NULL,
        PRIMARY KEY (searcher_id, column_index),
        FOREIGN KEY (searcher_id) REFERENCES neighbor_searchers(id) ON DELETE CASCADE
      )
    ''');

    // Store points (vectors) - chunked for large datasets
    db.execute('''
      CREATE TABLE IF NOT EXISTS searcher_points (
        searcher_id TEXT NOT NULL,
        point_index INTEGER NOT NULL,
        vector_data BLOB NOT NULL,
        bin_id INTEGER,
        PRIMARY KEY (searcher_id, point_index),
        FOREIGN KEY (searcher_id) REFERENCES neighbor_searchers(id) ON DELETE CASCADE
      )
    ''');

    // Add bin_id column if it doesn't exist (migration support)
    try {
      db.execute('ALTER TABLE searcher_points ADD COLUMN bin_id INTEGER');
    } catch (e) {
      // Column already exists, ignore
    }

    // Add text content columns for FTS (migration support)
    // Generic naming: source_text and target_text can be any languages
    try {
      db.execute('ALTER TABLE searcher_points ADD COLUMN source_text TEXT');
    } catch (e) {
      // Column already exists, ignore
    }
    try {
      db.execute('ALTER TABLE searcher_points ADD COLUMN target_text TEXT');
    } catch (e) {
      // Column already exists, ignore
    }
    // Migration: rename old columns if they exist
    try {
      db.execute('ALTER TABLE searcher_points ADD COLUMN french_text TEXT');
      // If old column exists, copy data and drop it
      db.execute('''
        UPDATE searcher_points 
        SET source_text = french_text 
        WHERE source_text IS NULL AND french_text IS NOT NULL
      ''');
    } catch (e) {
      // Column already exists or migration not needed, ignore
    }
    try {
      db.execute('ALTER TABLE searcher_points ADD COLUMN english_text TEXT');
      // If old column exists, copy data and drop it
      db.execute('''
        UPDATE searcher_points 
        SET target_text = english_text 
        WHERE target_text IS NULL AND english_text IS NOT NULL
      ''');
    } catch (e) {
      // Column already exists or migration not needed, ignore
    }

    // Create FTS virtual table for full-text search
    // Using standalone FTS table (not content table) for simplicity
    // We'll manually sync data to FTS
    try {
      db.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS searcher_points_fts USING fts5(
          searcher_id,
          point_index,
          source_text,
          target_text
        )
      ''');

      // Create index for fast lookups
      db.execute('''
        CREATE INDEX IF NOT EXISTS idx_fts_searcher_point
        ON searcher_points_fts(searcher_id, point_index)
      ''');
    } catch (e) {
      // Try FTS4 as fallback
      try {
        db.execute('''
          CREATE VIRTUAL TABLE IF NOT EXISTS searcher_points_fts USING fts4(
            searcher_id,
            point_index,
            source_text,
            target_text
          )
        ''');
      } catch (e2) {
        // FTS not available, continue without it
        print(
            'Warning: FTS not available in SQLite. Full-text search will be disabled.');
      }
    }

    // Store random projection vectors
    db.execute('''
      CREATE TABLE IF NOT EXISTS searcher_random_vectors (
        searcher_id TEXT NOT NULL,
        vector_index INTEGER NOT NULL,
        vector_data BLOB NOT NULL,
        PRIMARY KEY (searcher_id, vector_index),
        FOREIGN KEY (searcher_id) REFERENCES neighbor_searchers(id) ON DELETE CASCADE
      )
    ''');

    // Store bins (hash map structure)
    db.execute('''
      CREATE TABLE IF NOT EXISTS searcher_bins (
        searcher_id TEXT NOT NULL,
        bin_id INTEGER NOT NULL,
        point_index INTEGER NOT NULL,
        PRIMARY KEY (searcher_id, bin_id, point_index),
        FOREIGN KEY (searcher_id) REFERENCES neighbor_searchers(id) ON DELETE CASCADE
      )
    ''');

    // Indexes for performance
    db.execute('''
      CREATE INDEX IF NOT EXISTS idx_searcher_points 
      ON searcher_points(searcher_id, point_index)
    ''');
    db.execute('''
      CREATE INDEX IF NOT EXISTS idx_searcher_bins 
      ON searcher_bins(searcher_id, bin_id)
    ''');
    db.execute('''
      CREATE INDEX IF NOT EXISTS idx_searcher_random_vectors 
      ON searcher_random_vectors(searcher_id)
    ''');

    // Index for bin_id to support optimized queries
    db.execute('''
      CREATE INDEX IF NOT EXISTS idx_searcher_points_bin 
      ON searcher_points(searcher_id, bin_id)
    ''');
  }

  @override
  Future<String> saveSearcher(
    RandomBinaryProjectionSearcher searcher, {
    String? searcherId,
  }) async {
    if (searcherId != null && searcherId.isEmpty) {
      throw ArgumentError('searcherId cannot be empty if provided');
    }

    final db = _db!;
    final impl = searcher as RandomBinaryProjectionSearcherImpl;

    // Generate ID if not provided
    final id = searcherId ?? _generateId();

    // Start transaction
    db.execute('BEGIN TRANSACTION');

    try {
      // Extract metadata
      final digitCapacity = impl.digitCapacity;
      final seed = impl.seed;
      final schemaVersion = impl.schemaVersion;
      final dtype = impl.points.dtype;
      final columnCount = impl.points.columnCount;
      final pointCount = impl.points.rowCount;
      final createdAt = DateTime.now().toIso8601String();

      // Save metadata
      final metadataStmt = db.prepare('''
        INSERT OR REPLACE INTO neighbor_searchers 
        (id, digit_capacity, seed, schema_version, dtype, column_count, point_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      ''');
      metadataStmt.execute([
        id,
        digitCapacity,
        seed,
        schemaVersion,
        dtypeToString(dtype),
        columnCount,
        pointCount,
        createdAt,
      ]);
      metadataStmt.dispose();

      // Save column names
      final columnStmt = db.prepare('''
        INSERT OR REPLACE INTO searcher_columns 
        (searcher_id, column_index, column_name)
        VALUES (?, ?, ?)
      ''');
      var columnIndex = 0;
      for (final columnName in impl.columns) {
        columnStmt.execute([id, columnIndex, columnName]);
        columnIndex++;
      }
      columnStmt.dispose();

      // Save points (rows) with bin_id
      // Calculate bin IDs for all points
      final binIds = _calculateBinIds(impl.points, impl.randomVectors);

      final pointStmt = db.prepare('''
        INSERT OR REPLACE INTO searcher_points 
        (searcher_id, point_index, vector_data, bin_id)
        VALUES (?, ?, ?, ?)
      ''');
      for (var i = 0; i < impl.points.rowCount; i++) {
        final rowBlob = serializeMatrixRow(impl.points, i);
        final binId = binIds[i];
        pointStmt.execute([id, i, rowBlob, binId]);
      }
      pointStmt.dispose();

      // Save random vectors
      final randomVectorStmt = db.prepare('''
        INSERT OR REPLACE INTO searcher_random_vectors 
        (searcher_id, vector_index, vector_data)
        VALUES (?, ?, ?)
      ''');
      for (var i = 0; i < impl.randomVectors.rowCount; i++) {
        final rowBlob = serializeMatrixRow(impl.randomVectors, i);
        randomVectorStmt.execute([id, i, rowBlob]);
      }
      randomVectorStmt.dispose();

      // Save bins
      final binStmt = db.prepare('''
        INSERT OR REPLACE INTO searcher_bins 
        (searcher_id, bin_id, point_index)
        VALUES (?, ?, ?)
      ''');
      final flattenedBins = flattenBinMap(impl.bins);
      for (final entry in flattenedBins) {
        binStmt.execute([id, entry.key, entry.value]);
      }
      binStmt.dispose();

      // Commit transaction
      db.execute('COMMIT');

      return id;
    } catch (e) {
      // Rollback on error
      db.execute('ROLLBACK');
      rethrow;
    }
  }

  @override
  Future<RandomBinaryProjectionSearcher?> loadSearcher(
      String searcherId) async {
    final db = _db!;

    // Check if searcher exists
    final checkStmt =
        db.prepare('SELECT id FROM neighbor_searchers WHERE id = ?');
    final result = checkStmt.select([searcherId]);
    checkStmt.dispose();

    if (result.isEmpty) {
      return null;
    }

    // Load metadata
    final metadataStmt = db.prepare('''
      SELECT digit_capacity, seed, schema_version, dtype, column_count, point_count
      FROM neighbor_searchers
      WHERE id = ?
    ''');
    final metadataRow = metadataStmt.select([searcherId]).first;
    metadataStmt.dispose();

    final digitCapacity = metadataRow[0] as int;
    final seed = metadataRow[1] as int?;
    final schemaVersion = metadataRow[2] as int;
    final dtype = stringToDType(metadataRow[3] as String);
    //final columnCount = metadataRow[4] as int;
    final pointCount = metadataRow[5] as int;

    // Load column names
    final columnStmt = db.prepare('''
      SELECT column_name
      FROM searcher_columns
      WHERE searcher_id = ?
      ORDER BY column_index
    ''');
    final columnRows = columnStmt.select([searcherId]);
    final columns = columnRows.map((row) => row[0] as String).toList();
    columnStmt.dispose();

    // Load points
    final pointStmt = db.prepare('''
      SELECT vector_data
      FROM searcher_points
      WHERE searcher_id = ?
      ORDER BY point_index
    ''');
    final pointRows = pointStmt.select([searcherId]);
    final pointBlobs = pointRows.map((row) => row[0] as Uint8List).toList();
    pointStmt.dispose();

    if (pointBlobs.length != pointCount) {
      throw StateError(
          'Expected $pointCount points, but found ${pointBlobs.length}');
    }

    final points = deserializeMatrix(pointBlobs, dtype);

    // Load random vectors
    final randomVectorStmt = db.prepare('''
      SELECT vector_data
      FROM searcher_random_vectors
      WHERE searcher_id = ?
      ORDER BY vector_index
    ''');
    final randomVectorRows = randomVectorStmt.select([searcherId]);
    final randomVectorBlobs =
        randomVectorRows.map((row) => row[0] as Uint8List).toList();
    randomVectorStmt.dispose();

    final randomVectors = deserializeMatrix(randomVectorBlobs, dtype);

    // Load bins
    final binStmt = db.prepare('''
      SELECT bin_id, point_index
      FROM searcher_bins
      WHERE searcher_id = ?
      ORDER BY bin_id, point_index
    ''');
    final binRows = binStmt.select([searcherId]);
    final flattenedBins =
        binRows.map((row) => MapEntry(row[0] as int, row[1] as int)).toList();
    binStmt.dispose();

    final bins = reconstructBinMap(flattenedBins);

    // Reconstruct searcher
    final searcher = RandomBinaryProjectionSearcherImpl(
      columns,
      points,
      digitCapacity,
      seed: seed,
      schemaVersion: schemaVersion,
    );
    searcher.randomVectors = randomVectors;
    searcher.bins = bins;

    return searcher;
  }

  @override
  Future<bool> deleteSearcher(String searcherId) async {
    final db = _db!;

    // Check if searcher exists
    final checkStmt =
        db.prepare('SELECT id FROM neighbor_searchers WHERE id = ?');
    final result = checkStmt.select([searcherId]);
    checkStmt.dispose();

    if (result.isEmpty) {
      return false;
    }

    // Delete searcher (cascade will delete related records)
    final deleteStmt =
        db.prepare('DELETE FROM neighbor_searchers WHERE id = ?');
    deleteStmt.execute([searcherId]);
    deleteStmt.dispose();

    return true;
  }

  @override
  Future<List<String>> listSearchers() async {
    final db = _db!;

    final stmt =
        db.prepare('SELECT id FROM neighbor_searchers ORDER BY created_at');
    final rows = stmt.select([]);
    final ids = rows.map((row) => row[0] as String).toList();
    stmt.dispose();

    return ids;
  }

  @override
  Future<Map<String, dynamic>?> getSearcherMetadata(String searcherId) async {
    final db = _db!;

    // Load metadata
    final metadataStmt = db.prepare('''
      SELECT digit_capacity, seed, schema_version, dtype, column_count, point_count, created_at
      FROM neighbor_searchers
      WHERE id = ?
    ''');
    final rows = metadataStmt.select([searcherId]);
    metadataStmt.dispose();

    if (rows.isEmpty) {
      return null;
    }

    final row = rows.first;
    final digitCapacity = row[0] as int;
    final seed = row[1] as int?;
    final schemaVersion = row[2] as int;
    final dtype = row[3] as String;
    final columnCount = row[4] as int;
    final pointCount = row[5] as int;
    final createdAt = row[6] as String;

    // Load column names
    final columnStmt = db.prepare('''
      SELECT column_name
      FROM searcher_columns
      WHERE searcher_id = ?
      ORDER BY column_index
    ''');
    final columnRows = columnStmt.select([searcherId]);
    final columns = columnRows.map((row) => row[0] as String).toList();
    columnStmt.dispose();

    return {
      'digitCapacity': digitCapacity,
      'seed': seed,
      'schemaVersion': schemaVersion,
      'dtype': dtype,
      'columnCount': columnCount,
      'pointCount': pointCount,
      'createdAt': createdAt,
      'columns': columns,
    };
  }

  /// Closes the database connection.
  ///
  /// Call this when you're done using the store to free resources.
  void close() {
    _db?.dispose();
    _db = null;
  }

  @override
  Future<DataFrame?> loadSearcherData(String searcherId) async {
    if (searcherId.isEmpty) {
      throw ArgumentError('searcherId cannot be empty');
    }

    final db = _db!;

    // Check if searcher exists
    final checkStmt =
        db.prepare('SELECT id FROM neighbor_searchers WHERE id = ?');
    final result = checkStmt.select([searcherId]);
    checkStmt.dispose();

    if (result.isEmpty) {
      return null;
    }

    // Load column names
    final columnStmt = db.prepare('''
      SELECT column_name
      FROM searcher_columns
      WHERE searcher_id = ?
      ORDER BY column_index
    ''');
    final columnRows = columnStmt.select([searcherId]);
    final columns = columnRows.map((row) => row[0] as String).toList();
    columnStmt.dispose();

    // Load points
    final pointStmt = db.prepare('''
      SELECT vector_data
      FROM searcher_points
      WHERE searcher_id = ?
      ORDER BY point_index
    ''');
    final pointRows = pointStmt.select([searcherId]);
    final pointBlobs = pointRows.map((row) => row[0] as Uint8List).toList();
    pointStmt.dispose();

    if (pointBlobs.isEmpty) {
      return null;
    }

    // Deserialize all rows
    // Note: deserializeMatrixRow reads dtype from each blob internally
    final rows = <List<num>>[];
    for (final blob in pointBlobs) {
      final rowValues = deserializeMatrixRow(blob);
      rows.add(rowValues);
    }

    // Create DataFrame
    return DataFrame(rows, header: columns);
  }

  /// Trains a new searcher from data stored in a SQLite table.
  ///
  /// This method allows you to train a searcher directly from data stored in your
  /// own SQLite tables, enabling retraining workflows without exporting to CSV/JSON.
  ///
  /// [tableName] is the name of the table containing the data.
  /// [embeddingColumns] is a list of column names that contain the embedding vectors.
  /// [whereClause] is an optional WHERE clause to filter the data (without the WHERE keyword).
  /// [whereArgs] are optional arguments for the WHERE clause.
  ///
  /// **Security Note**: Column names and table names are validated to prevent SQL injection.
  /// However, [whereClause] should only contain trusted input or use parameterized queries.
  ///
  /// Example:
  ///
  /// ```dart
  /// // Train from all phrases in translations table
  /// final searcher = await store.trainFromTable(
  ///   'translations',
  ///   ['embedding_0', 'embedding_1', 'embedding_2', ...],
  ///   digitCapacity: 8,
  /// );
  ///
  /// // Train from specific language pairs only
  /// final searcher = await store.trainFromTable(
  ///   'translations',
  ///   ['embedding_0', 'embedding_1', 'embedding_2', ...],
  ///   digitCapacity: 8,
  ///   whereClause: 'source_lang = ? AND target_lang = ?',
  ///   whereArgs: ['en', 'fr'],
  /// );
  /// ```
  Future<RandomBinaryProjectionSearcher> trainFromTable(
    String tableName,
    List<String> embeddingColumns, {
    required int digitCapacity,
    int? seed,
    DType dtype = DType.float32,
    String? whereClause,
    List<Object>? whereArgs,
  }) async {
    final db = _db!;

    // Validate inputs
    if (embeddingColumns.isEmpty) {
      throw ArgumentError('embeddingColumns cannot be empty');
    }

    // Validate table and column names to prevent SQL injection
    _validateIdentifier(tableName, 'tableName');
    for (var i = 0; i < embeddingColumns.length; i++) {
      _validateIdentifier(embeddingColumns[i], 'embeddingColumns[$i]');
    }

    // Build query with validated identifiers
    // Note: SQLite doesn't support parameterized column/table names, so we validate them
    final columnList = embeddingColumns.map((col) => '"$col"').join(', ');
    var query = 'SELECT $columnList FROM "$tableName"';
    if (whereClause != null) {
      // whereClause should use parameterized queries (?) for values
      query += ' WHERE $whereClause';
    }

    final stmt = db.prepare(query);
    final rows = whereArgs != null ? stmt.select(whereArgs) : stmt.select([]);

    // Convert to DataFrame
    final dataRows = <List<num>>[];
    for (final row in rows) {
      final dataRow = <num>[];
      for (var i = 0; i < embeddingColumns.length; i++) {
        final value = row[i];
        if (value is num) {
          dataRow.add(value);
        } else if (value is String) {
          dataRow.add(double.parse(value));
        } else {
          throw ArgumentError(
              'Column ${embeddingColumns[i]} contains non-numeric value: $value');
        }
      }
      dataRows.add(dataRow);
    }
    stmt.dispose();

    if (dataRows.isEmpty) {
      throw StateError(
          'No data found in table "$tableName"${whereClause != null ? " matching WHERE clause" : ""}');
    }

    // Create DataFrame and train searcher
    final data = DataFrame(dataRows, headerExists: false);
    return RandomBinaryProjectionSearcher(
      data,
      digitCapacity,
      seed: seed,
      dtype: dtype,
    );
  }

  /// Retrains a searcher from its stored data.
  ///
  /// This is a convenience method that loads the data from an existing searcher,
  /// retrains it with new parameters, and optionally saves it with a new ID.
  ///
  /// [searcherId] is the ID of the searcher to retrain.
  /// [digitCapacity] is the new digit capacity (can be different from original).
  /// [seed] is an optional seed for the new searcher.
  /// [dtype] is the data type for the new searcher.
  ///
  /// Returns the newly trained searcher.
  ///
  /// Example:
  ///
  /// ```dart
  /// // Retrain with same parameters
  /// final retrained = await store.retrainSearcher('old-searcher-id', digitCapacity: 8);
  ///
  /// // Retrain with different parameters
  /// final retrained = await store.retrainSearcher(
  ///   'old-searcher-id',
  ///   digitCapacity: 10,
  ///   seed: 999,
  /// );
  ///
  /// // Save retrained searcher
  /// final newId = await retrained.saveToStore(store, searcherId: 'new-searcher-id');
  /// ```
  Future<RandomBinaryProjectionSearcher> retrainSearcher(
    String searcherId, {
    required int digitCapacity,
    int? seed,
    DType dtype = DType.float32,
  }) async {
    final data = await loadSearcherData(searcherId);
    if (data == null) {
      throw ArgumentError('Searcher with ID $searcherId not found');
    }

    return RandomBinaryProjectionSearcher(
      data,
      digitCapacity,
      seed: seed,
      dtype: dtype,
    );
  }

  /// Validates that a string is a valid SQLite identifier.
  ///
  /// SQLite identifiers must:
  /// - Start with a letter or underscore
  /// - Contain only letters, digits, and underscores
  /// - Not be empty
  ///
  /// This helps prevent SQL injection when constructing queries.
  void _validateIdentifier(String identifier, String parameterName) {
    if (identifier.isEmpty) {
      throw ArgumentError('$parameterName cannot be empty');
    }

    // SQLite identifier regex: must start with letter/underscore, then letters/digits/underscores
    final identifierPattern = RegExp(r'^[a-zA-Z_][a-zA-Z0-9_]*$');
    if (!identifierPattern.hasMatch(identifier)) {
      throw ArgumentError(
          '$parameterName must be a valid SQLite identifier (letters, digits, underscores only, starting with letter/underscore): "$identifier"');
    }
  }

  String _generateId() {
    // Generate a unique ID using timestamp and random component
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final random = (timestamp % 1000000).toString().padLeft(6, '0');
    return 'searcher_${timestamp}_$random';
  }

  /// Calculates bin IDs for all points in the matrix.
  ///
  /// This is used when saving searchers to store bin_id in the database.
  List<int> _calculateBinIds(Matrix points, Matrix randomVectors) {
    final binaryRep = getBinaryRepresentation(points, randomVectors);
    final binIds = getBinIdsFromBinaryRepresentation(binaryRep);
    return binIds.toList();
  }

  /// Loads an optimized searcher that uses lazy loading and two-phase filtering.
  ///
  /// This method loads only metadata (bins, random vectors) into memory,
  /// and uses SQLite for on-demand point loading during queries.
  ///
  /// [searcherId] is the ID of the searcher to load.
  /// [approximateFilterLimit] is the maximum number of candidates to load
  ///   for exact distance calculation (default: 2000).
  /// [cacheSize] is the maximum number of points to cache in memory (default: 1000).
  ///
  /// Returns `null` if the searcher doesn't exist.
  ///
  /// Example:
  ///
  /// ```dart
  /// final optimized = await store.loadOptimizedSearcher('my-searcher-id');
  /// final neighbours = optimized!.query(point, k, searchRadius);
  /// ```
  Future<SQLiteOptimizedSearcher?> loadOptimizedSearcher(
    String searcherId, {
    int approximateFilterLimit = 2000,
    int cacheSize = 1000,
  }) async {
    final db = _db!;

    // Check if searcher exists
    final checkStmt =
        db.prepare('SELECT id FROM neighbor_searchers WHERE id = ?');
    final result = checkStmt.select([searcherId]);
    checkStmt.dispose();

    if (result.isEmpty) {
      return null;
    }

    // Load metadata
    final metadataStmt = db.prepare('''
      SELECT digit_capacity, seed, schema_version, dtype, column_count, point_count
      FROM neighbor_searchers
      WHERE id = ?
    ''');
    final metadataRow = metadataStmt.select([searcherId]).first;
    metadataStmt.dispose();

    final digitCapacity = metadataRow[0] as int;
    final seed = metadataRow[1] as int?;
    final schemaVersion = metadataRow[2] as int;
    final dtype = stringToDType(metadataRow[3] as String);
    final columnCount = metadataRow[4] as int;
    final pointCount = metadataRow[5] as int;

    // Load column names
    final columnStmt = db.prepare('''
      SELECT column_name
      FROM searcher_columns
      WHERE searcher_id = ?
      ORDER BY column_index
    ''');
    final columnRows = columnStmt.select([searcherId]);
    final columns = columnRows.map((row) => row[0] as String).toList();
    columnStmt.dispose();

    // Load random vectors (small, keep in memory)
    final randomVectorStmt = db.prepare('''
      SELECT vector_data
      FROM searcher_random_vectors
      WHERE searcher_id = ?
      ORDER BY vector_index
    ''');
    final randomVectorRows = randomVectorStmt.select([searcherId]);
    final randomVectorBlobs =
        randomVectorRows.map((row) => row[0] as Uint8List).toList();
    randomVectorStmt.dispose();

    final randomVectors = deserializeMatrix(randomVectorBlobs, dtype);

    // Load bins (small, keep in memory)
    final binStmt = db.prepare('''
      SELECT bin_id, point_index
      FROM searcher_bins
      WHERE searcher_id = ?
      ORDER BY bin_id, point_index
    ''');
    final binRows = binStmt.select([searcherId]);
    final flattenedBins =
        binRows.map((row) => MapEntry(row[0] as int, row[1] as int)).toList();
    binStmt.dispose();

    final bins = reconstructBinMap(flattenedBins);

    // Create lazy point accessor
    final pointAccessor = SQLiteLazyPointAccessor(
      db,
      searcherId,
      dtype,
      columnCount,
      pointCount,
      maxCacheSize: cacheSize,
    );

    // Create optimized searcher
    return SQLiteOptimizedSearcher(
      columns: columns,
      pointAccessor: pointAccessor,
      randomVectors: randomVectors,
      bins: bins,
      digitCapacity: digitCapacity,
      seed: seed,
      schemaVersion: schemaVersion,
      db: db,
      searcherId: searcherId,
      approximateFilterLimit: approximateFilterLimit,
    );
  }

  /// Adds text content (source and target) to a point for FTS indexing.
  ///
  /// This enables full-text search on translation pairs.
  /// Generic naming: source and target can be any languages.
  Future<void> addTextContent(
    String searcherId,
    int pointIndex, {
    required String sourceText,
    required String targetText,
  }) async {
    final db = _db!;

    // Update searcher_points table
    final updateStmt = db.prepare('''
      UPDATE searcher_points
      SET source_text = ?, target_text = ?
      WHERE searcher_id = ? AND point_index = ?
    ''');

    try {
      updateStmt.execute([sourceText, targetText, searcherId, pointIndex]);

      // Update FTS index (if available)
      try {
        // Delete old entry if exists
        final deleteStmt = db.prepare('''
          DELETE FROM searcher_points_fts
          WHERE searcher_id = ? AND point_index = ?
        ''');
        deleteStmt.execute([searcherId, pointIndex]);
        deleteStmt.dispose();

        // Insert new entry
        final ftsStmt = db.prepare('''
          INSERT INTO searcher_points_fts(searcher_id, point_index, source_text, target_text)
          VALUES (?, ?, ?, ?)
        ''');
        ftsStmt.execute([searcherId, pointIndex, sourceText, targetText]);
        ftsStmt.dispose();
      } catch (e) {
        // FTS table might not exist, ignore
      }
    } finally {
      updateStmt.dispose();
    }
  }

  /// Performs FTS search and returns matching point indices.
  ///
  /// Returns empty list if FTS is not available or no matches found.
  List<int> ftsSearch(String searcherId, String keywordQuery) {
    final db = _db!;
    final results = <int>[];

    try {
      // Escape special FTS characters and format for FTS5
      // FTS5 query syntax: for multi-word, use quotes or space-separated
      // Escape single quotes and wrap in quotes for phrase search
      var escapedQuery = keywordQuery.replaceAll("'", "''");

      // For FTS5, if query has multiple words, we can search each word
      // or use phrase search with quotes. Let's use phrase search for exact match
      // and also try individual words as fallback
      final words = keywordQuery.trim().split(RegExp(r'\s+'));

      // Build FTS5 query: use phrase search (quoted) for exact match
      // FTS5 supports: "exact phrase" or word1 word2 (OR) or word1 OR word2
      final phraseQuery = '"$escapedQuery"';

      // FTS5 query: search in source_text and target_text columns
      // Try phrase search first, then fallback to word search
      final ftsStmt = db.prepare('''
        SELECT point_index FROM searcher_points_fts
        WHERE searcher_id = ?
          AND (source_text MATCH ? OR target_text MATCH ?)
      ''');

      // Try phrase search first (exact phrase match)
      var rows = ftsStmt.select([searcherId, phraseQuery, phraseQuery]);

      // If no results, try word-based search (any word matches)
      if (rows.isEmpty && words.length > 1) {
        final escapedWordQuery =
            words.map((w) => w.replaceAll("'", "''")).join(' OR ');
        rows = ftsStmt.select([searcherId, escapedWordQuery, escapedWordQuery]);
      }

      // If still no results, try simple query (single word or exact match)
      if (rows.isEmpty) {
        rows = ftsStmt.select([searcherId, escapedQuery, escapedQuery]);
      }

      for (final row in rows) {
        results.add(row[0] as int);
      }

      ftsStmt.dispose();
    } catch (e) {
      // FTS not available or query failed, return empty
      // For debugging: print('FTS error: $e');
      return [];
    }

    return results;
  }

  /// Gets text content for a point.
  Future<TextContent> getTextContent(
    String searcherId,
    int pointIndex,
  ) async {
    final db = _db!;
    final stmt = db.prepare('''
      SELECT source_text, target_text
      FROM searcher_points
      WHERE searcher_id = ? AND point_index = ?
    ''');

    try {
      final rows = stmt.select([searcherId, pointIndex]);
      if (rows.isEmpty) {
        return TextContent(null, null);
      }

      final row = rows.first;
      return TextContent(
        row[0] as String?,
        row[1] as String?,
      );
    } finally {
      stmt.dispose();
    }
  }
}
