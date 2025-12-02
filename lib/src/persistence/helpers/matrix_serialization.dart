import 'dart:typed_data';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

/// Serializes a matrix row to a BLOB for SQLite storage.
///
/// The format is:
/// - 4 bytes: column count (int32)
/// - 1 byte: dtype (0 = float32, 1 = float64)
/// - N bytes: row data (float32 = 4 bytes per element, float64 = 8 bytes per element)
Uint8List serializeMatrixRow(Matrix matrix, int rowIndex) {
  final row = matrix[rowIndex];
  final rowValues = row.toList(); // Convert Vector to List for safe access
  final columnCount = matrix.columnCount;
  final dtype = matrix.dtype;

  final elementSize = dtype == DType.float32 ? 4 : 8;
  final dataSize = columnCount * elementSize;
  final totalSize = 4 + 1 + dataSize; // columnCount + dtype + data

  final buffer = Uint8List(totalSize);
  final byteData = ByteData.view(buffer.buffer);

  // Write column count
  byteData.setInt32(0, columnCount, Endian.little);

  // Write dtype (0 = float32, 1 = float64)
  byteData.setUint8(4, dtype == DType.float32 ? 0 : 1);

  // Write row data
  final dataOffset = 5;
  if (dtype == DType.float32) {
    for (var i = 0; i < columnCount; i++) {
      byteData.setFloat32(
          dataOffset + i * 4, rowValues[i].toDouble(), Endian.little);
    }
  } else {
    for (var i = 0; i < columnCount; i++) {
      byteData.setFloat64(
          dataOffset + i * 8, rowValues[i].toDouble(), Endian.little);
    }
  }

  return buffer;
}

/// Deserializes a matrix row from a BLOB.
///
/// Returns a list of doubles representing the row values.
List<double> deserializeMatrixRow(Uint8List blob) {
  final byteData = ByteData.view(blob.buffer);

  // Read column count
  final columnCount = byteData.getInt32(0, Endian.little);

  // Read dtype
  final dtypeValue = byteData.getUint8(4);
  final dtype = dtypeValue == 0 ? DType.float32 : DType.float64;

  // Read row data
  final dataOffset = 5;
  final row = <double>[];

  if (dtype == DType.float32) {
    for (var i = 0; i < columnCount; i++) {
      row.add(byteData.getFloat32(dataOffset + i * 4, Endian.little));
    }
  } else {
    for (var i = 0; i < columnCount; i++) {
      row.add(byteData.getFloat64(dataOffset + i * 8, Endian.little));
    }
  }

  return row;
}

/// Serializes a full matrix to a list of BLOBs (one per row).
List<Uint8List> serializeMatrix(Matrix matrix) {
  final rows = <Uint8List>[];
  for (var i = 0; i < matrix.rowCount; i++) {
    rows.add(serializeMatrixRow(matrix, i));
  }
  return rows;
}

/// Deserializes a matrix from a list of BLOBs.
///
/// The rows are expected to be in order (row 0, row 1, ...).
Matrix deserializeMatrix(List<Uint8List> rowBlobs, DType dtype) {
  if (rowBlobs.isEmpty) {
    throw ArgumentError('Cannot deserialize empty matrix');
  }

  // Read first row to get column count
  final firstRow = deserializeMatrixRow(rowBlobs[0]);
  final columnCount = firstRow.length;
  final rows = <List<double>>[firstRow];

  // Read remaining rows
  for (var i = 1; i < rowBlobs.length; i++) {
    final row = deserializeMatrixRow(rowBlobs[i]);
    if (row.length != columnCount) {
      throw ArgumentError(
          'Row $i has ${row.length} columns, expected $columnCount');
    }
    rows.add(row);
  }

  return Matrix.fromList(rows, dtype: dtype);
}
