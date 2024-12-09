import 'package:flutter/material.dart';
import 'package:test_onnxruntime_flutter/prediction.dart';

void main() async {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Hello World!'),
              ElevatedButton(
                onPressed: predict,
                child: Text('Predict'),
              )
            ],
          ),
        ),
      ),
    );
  }
}

void predict() async {
  const input = AgentInput(
    board: [
      [1, 2, 1, 2, 3, 3, 3],
      [2, 0, 2, 1, 3, 3, 3],
      [1, 1, 2, 2, 3, 3, 3],
      [2, 1, 2, 1, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3]
    ],
    orbitDirection: 1,
  );

  const assetFileName = 'assets/models/best_model.onnx'; // optset_version=18
  final gameAgent = GameAgent(modelFileName: assetFileName);
  final output = await gameAgent.predict(input);
  for (var e in output!) {
    print(e!.value);
  }
}
