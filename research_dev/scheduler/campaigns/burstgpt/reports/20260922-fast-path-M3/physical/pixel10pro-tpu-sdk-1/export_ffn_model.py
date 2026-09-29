"""Export a static real-weight SwiGLU FFN probe and independent FP32 references."""

import argparse
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("activation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.batch not in (1, 2, 4, 8):
        parser.error("batch must be 1, 2, 4, or 8")
    args.output.mkdir(parents=True, exist_ok=False)
    weights = {k: v.astype(np.float32) for k, v in np.load(args.weights).items()}
    width, embedding = weights["gate"].shape
    if weights["up"].shape != (width, embedding) or weights["down"].shape != (embedding, width):
        raise ValueError("inconsistent FFN matrices")
    model = s.ModelT()
    model.version = 3
    model.description = "Qwen3 real F16 weights expanded to FP32, gate * sigmoid(gate) * up, then down"
    model.buffers = [s.BufferT()]
    graph = s.SubGraphT()
    graph.name = "main"
    graph.tensors = []
    model.subgraphs = [graph]

    def tensor(name, shape, data=None):
        value = s.TensorT()
        value.name, value.shape, value.type, value.hasRank = name, shape, s.TensorType.FLOAT32, True
        if data is not None:
            buffer = s.BufferT()
            buffer.data = np.frombuffer(data.astype("<f4").tobytes(), dtype=np.uint8)
            value.buffer = len(model.buffers)
            model.buffers.append(buffer)
        graph.tensors.append(value)
        return len(graph.tensors)-1

    x = tensor("input", [args.batch, embedding])
    wg = tensor("gate_weight", [width, embedding], weights["gate"])
    wu = tensor("up_weight", [width, embedding], weights["up"])
    wd = tensor("down_weight", [embedding, width], weights["down"])
    gate, up, sigmoid, silu, product = [tensor(n, [args.batch, width]) for n in
                                      ("gate", "up", "sigmoid", "silu", "product")]
    out = tensor("output", [args.batch, embedding])
    model.operatorCodes = []
    graph.operators = []
    definitions = [(s.BuiltinOperator.FULLY_CONNECTED, s.BuiltinOptions.FullyConnectedOptions, s.FullyConnectedOptionsT),
                   (s.BuiltinOperator.LOGISTIC, s.BuiltinOptions.NONE, None),
                   (s.BuiltinOperator.MUL, s.BuiltinOptions.MulOptions, s.MulOptionsT)]
    for code, _, _ in definitions:
        opcode = s.OperatorCodeT()
        opcode.builtinCode = opcode.deprecatedBuiltinCode = code
        opcode.version = 1
        model.operatorCodes.append(opcode)
    for index, inputs, output in [(0, [x, wg, -1], gate), (0, [x, wu, -1], up),
                                  (1, [gate], sigmoid), (2, [gate, sigmoid], silu),
                                  (2, [silu, up], product), (0, [product, wd, -1], out)]:
        op = s.OperatorT()
        op.opcodeIndex, op.inputs, op.outputs = index, inputs, [output]
        op.builtinOptionsType = definitions[index][1]
        factory = definitions[index][2]
        op.builtinOptions = factory() if factory else None
        graph.operators.append(op)
    graph.inputs, graph.outputs = [x], [out]
    signature = s.SignatureDefT()
    signature.signatureKey = "serving_default"
    signature.inputs, signature.outputs = [], []
    for name, index, mappings in [("input", x, signature.inputs), ("output", out, signature.outputs)]:
        mapping = s.TensorMapT()
        mapping.name, mapping.tensorIndex = name, index
        mappings.append(mapping)
    model.signatureDefs = [signature]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    dest = args.output / "ffn.tflite"
    dest.write_bytes(builder.Output())
    original = np.fromfile(args.activation, dtype="<f2").astype(np.float32).reshape(1, embedding)
    rng = np.random.default_rng(20260924)
    cases = [original, original * 0.5, original * 1.5, -original]
    cases += [rng.normal(0, original.std(), size=original.shape).astype(np.float16).astype(np.float32)
              for _ in range(4)]
    inputs = np.stack([np.concatenate([cases[(i+j) % len(cases)] for j in range(args.batch)])
                       for i in range(len(cases))]).astype("<f4")
    g = inputs @ weights["gate"].T
    u = inputs @ weights["up"].T
    reference = (g / (1.0 + np.exp(-g)) * u) @ weights["down"].T
    interpreter = Interpreter(model_path=str(dest), num_threads=2)
    interpreter.allocate_tensors()
    results = []
    for case in inputs:
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], case)
        interpreter.invoke()
        results.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]))
    actual = np.stack(results)
    error = np.linalg.norm((actual-reference).astype(np.float64).reshape(len(cases), -1), axis=1)
    error /= np.linalg.norm(reference.astype(np.float64).reshape(len(cases), -1), axis=1)
    np.savez(args.output / "vectors.npz", inputs=inputs, reference=reference, tflite_cpu=actual)
    record = dict(status="PASS" if max(error) < 0.0001 else "FAIL", width=width, embedding=embedding,
                  batch=args.batch, cases=len(cases), ops=6, max_cpu_vs_numpy_relative_l2=float(max(error)),
                  activation_sha256=hashlib.sha256(args.activation.read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  formula="((x @ gate.T) * sigmoid(x @ gate.T) * (x @ up.T)) @ down.T")
    (args.output / "MODEL.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record["status"] != "PASS":
        raise RuntimeError("TFLite CPU reference mismatch")


if __name__ == "__main__":
    main()
