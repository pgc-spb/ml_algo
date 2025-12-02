import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/persistence/sqlite_neighbor_search_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:sqlite3/sqlite3.dart' show sqlite3;
import 'package:test/test.dart';

void main() {
  group('SQLite Retraining', () {
    late String testDbPath;
    late SQLiteNeighborSearchStore store;

    setUp(() {
      testDbPath =
          'test_retraining_${DateTime.now().millisecondsSinceEpoch}.db';
      store = SQLiteNeighborSearchStore(testDbPath);
    });

    tearDown(() async {
      store.close();
      final file = File(testDbPath);
      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should load searcher data as DataFrame', () async {
      final data = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4, seed: 10);

      final searcherId = await searcher.saveToStore(store);
      final loadedData = await store.loadSearcherData(searcherId);

      expect(loadedData, isNotNull);
      expect(loadedData!.rows.length, 3);
      expect(loadedData.header.length, 3);
      expect(loadedData.header, data.header);
    });

    test('should retrain searcher from stored data', () async {
      final originalData = DataFrame([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ], headerExists: false);
      final originalSearcher = RandomBinaryProjectionSearcher(
        originalData,
        4,
        seed: 10,
      );

      final searcherId = await originalSearcher.saveToStore(store);

      // Retrain with different parameters
      final retrained = await store.retrainSearcher(
        searcherId,
        digitCapacity: 6,
        seed: 20,
      );

      expect(retrained.digitCapacity, 6);
      expect(retrained.seed, 20);
      expect(retrained.points.rowCount, 3);
      expect(retrained.points.columnCount, 3);

      // Verify query still works
      final queryPoint = Vector.fromList([2, 3, 4]);
      final neighbours = retrained.query(queryPoint, 2, 2);
      expect(neighbours.length, 2);
    });

    test('should retrain using trainFromStore static method', () async {
      final data = DataFrame([
        [10, 20, 30],
        [40, 50, 60],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(data, 4);

      final searcherId = await searcher.saveToStore(store);

      // Retrain using static method
      final retrained = await RandomBinaryProjectionSearcher.trainFromStore(
        store,
        searcherId,
        digitCapacity: 8,
        seed: 99,
      );

      expect(retrained.digitCapacity, 8);
      expect(retrained.seed, 99);
      expect(retrained.points.rowCount, 2);
    });

    test('should train from custom SQLite table', () async {
      // Access database through a helper - we'll use sqlite3 directly
      final db = sqlite3.open(testDbPath);

      // Create a custom table with phrase embeddings
      db.execute('''
        CREATE TABLE IF NOT EXISTS phrase_embeddings (
          id INTEGER PRIMARY KEY,
          phrase TEXT,
          embedding_0 REAL,
          embedding_1 REAL,
          embedding_2 REAL,
          language TEXT
        )
      ''');

      // Insert test data
      final insertStmt = db.prepare('''
        INSERT INTO phrase_embeddings 
        (phrase, embedding_0, embedding_1, embedding_2, language)
        VALUES (?, ?, ?, ?, ?)
      ''');

      insertStmt.execute(['hello', 0.1, 0.2, 0.3, 'en']);
      insertStmt.execute(['world', 0.4, 0.5, 0.6, 'en']);
      insertStmt.execute(['bonjour', 0.7, 0.8, 0.9, 'fr']);
      insertStmt.dispose();

      // Train searcher from table
      final searcher = await store.trainFromTable(
        'phrase_embeddings',
        ['embedding_0', 'embedding_1', 'embedding_2'],
        digitCapacity: 4,
        seed: 42,
      );

      expect(searcher.points.rowCount, 3);
      expect(searcher.points.columnCount, 3);

      // Query
      final queryPoint = Vector.fromList([0.2, 0.3, 0.4]);
      final neighbours = searcher.query(queryPoint, 2, 2);
      expect(neighbours.length, 2);
    });

    test('should train from SQLite table with WHERE clause', () async {
      final db = sqlite3.open(testDbPath);

      // Create a custom table
      db.execute('''
        CREATE TABLE IF NOT EXISTS translations (
          id INTEGER PRIMARY KEY,
          source_text TEXT,
          target_text TEXT,
          embedding_0 REAL,
          embedding_1 REAL,
          embedding_2 REAL,
          source_lang TEXT,
          target_lang TEXT
        )
      ''');

      // Insert test data
      final insertStmt = db.prepare('''
        INSERT INTO translations 
        (source_text, target_text, embedding_0, embedding_1, embedding_2, source_lang, target_lang)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      ''');

      insertStmt.execute(['hello', 'bonjour', 0.1, 0.2, 0.3, 'en', 'fr']);
      insertStmt.execute(['world', 'monde', 0.4, 0.5, 0.6, 'en', 'fr']);
      insertStmt.execute(['hello', 'hola', 0.7, 0.8, 0.9, 'en', 'es']);
      insertStmt.dispose();
      db.dispose();

      // Train searcher from table with WHERE clause (only en->fr)
      final searcher = await store.trainFromTable(
        'translations',
        ['embedding_0', 'embedding_1', 'embedding_2'],
        digitCapacity: 4,
        whereClause: 'source_lang = ? AND target_lang = ?',
        whereArgs: ['en', 'fr'],
      );

      expect(searcher.points.rowCount, 2); // Only en->fr pairs

      // Query
      final queryPoint = Vector.fromList([0.2, 0.3, 0.4]);
      final neighbours = searcher.query(queryPoint, 2, 2);
      expect(neighbours.length, 2);
    });

    test('should handle retraining with different dtype', () async {
      final data = DataFrame([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ], headerExists: false);
      final searcher = RandomBinaryProjectionSearcher(
        data,
        4,
        dtype: DType.float32,
      );

      final searcherId = await searcher.saveToStore(store);

      // Retrain with float64
      final retrained = await store.retrainSearcher(
        searcherId,
        digitCapacity: 4,
        dtype: DType.float64,
      );

      expect(retrained.points.dtype, DType.float64);
    });

    test('should throw error when retraining non-existent searcher', () async {
      expect(
        () => store.retrainSearcher('non-existent', digitCapacity: 4),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('should throw error when training from empty table', () async {
      final db = sqlite3.open(testDbPath);

      db.execute('''
        CREATE TABLE IF NOT EXISTS empty_table (
          id INTEGER PRIMARY KEY,
          embedding_0 REAL,
          embedding_1 REAL
        )
      ''');
      db.dispose();

      expect(
        () => store.trainFromTable(
          'empty_table',
          ['embedding_0', 'embedding_1'],
          digitCapacity: 4,
        ),
        throwsA(isA<StateError>()),
      );
    });
  });
}
