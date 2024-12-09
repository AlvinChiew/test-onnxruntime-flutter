import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class AgentInput {
  final List<List<int>> board;
  final int orbitDirection;

  const AgentInput({
    required this.board,
    required this.orbitDirection,
  });

  Map<String, OrtValue> convertToOrtTensor() {
    // Flatten the board and cast to int8
    final flattenedBoard = board.expand((row) => row).toList();
    final boardInt8 = Int8List.fromList(flattenedBoard);

    // Add an extra dimension for batch size (e.g., shape: [1, rows, cols])
    final boardTensor = OrtValueTensor.createTensorWithDataList(
      boardInt8,
      [1, board.length, board[0].length], // Add batch dimension
    );

    // Convert orbitDirection to int8 tensor
    final orbitDirectionInt8 = Int8List.fromList([orbitDirection]);
    final orbitDirectionTensor = OrtValueTensor.createTensorWithDataList(
      orbitDirectionInt8,
      [1],
    );

    return {
      'board': boardTensor,
      'orbit_direction': orbitDirectionTensor,
    };
  }
}

class GameAgent {
  final String modelFileName;

  OrtSession? session;
  List<OrtValue?>? output;
  final runOptions = OrtRunOptions();

  GameAgent({
    required this.modelFileName,
  });

  Future<void> createSession() async {
    OrtEnv.instance.init();

    final sessionOptions = OrtSessionOptions();
    final rawAssetFile = await rootBundle.load(modelFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    session = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  void releaseSession() {
    runOptions.release();
    output?.forEach((element) {
      element?.release();
    });

    OrtEnv.instance.release();
  }

  Future<List<OrtValue?>?> predict(AgentInput input) async {
    if (session == null) {
      await createSession();
    }
    output = await session!.runAsync(runOptions, input.convertToOrtTensor());
    return output;
  }
}
