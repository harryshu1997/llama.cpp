"""Export one Qwen3-32B Q4_0 SwiGLU FFN layer as a static-shape TFLite graph for the Tensor G5 compiler.

y = (silu(x @ gate.T) * (x @ up.T)) @ down.T, x/y FP32 [M, 5120], intermediate [M, 25600].

Weight representations (--weights):
  f32       FP32 constants = exact d*q (compiler then truncates to fp16/bf16).        6 ops
  f16       FLOAT16 constants -> DEQUANTIZE -> FC (standard TFLite fp16 format).      9 ops
  i4bw      INT4 [out,in] + BlockwiseQuantization(block 32 along in, FLOAT16 scale
            tensor [out, in/32] = the Q4_0 d values verbatim, zero point none);
            FC consumes the int4 tensor directly (TFLite hybrid / DRQ form).          6 ops
  i4bw_dq   same INT4 blockwise tensor -> DEQUANTIZE -> FC (weight-only form).        9 ops
  i4bw_pos  INT4 blockwise with |d| scales: q flipped in negative-d blocks and clamped (+8 -> 7).
            NOT Q4_0-exact (diagnostic for the scale-sign question).
  i4bw_zp   INT4 blockwise with |d| scales and per-block zero point (0, or -1 in negative-d blocks
            with q' = -q - 1): exact in real arithmetic if the zero points are honoured.
  i4emu     "emulated sub-channel" TFLite pattern (ai-edge-quantizer <=0.1 EMULATED_SUBCHANNEL):
            x[M,K] -> RESHAPE [M,K/32,1,32] -> BATCH_MATMUL with INT4 [1,K/32,32,N] (per-tensor
            scale 1, q verbatim) -> MUL by FP32 scales [1,K/32,1,N] (= Q4_0 d, signed) ->
            SUM(axis 1) -> RESHAPE [M,N]. Exact Q4_0 representation.
  i4wo      weight-only dequant pattern: INT4 [N,K/32,32] (scale 1, q verbatim) -> DEQUANTIZE ->
            MUL by FP16->FP32 scales [N,K/32,1] (Q4_0 d) -> RESHAPE [N,K] -> FC. Exact Q4_0.
  i4pc      INT4 per-output-channel symmetric requantization of the dequantized weights. NOT exact.
  i8pc      INT8 per-output-channel symmetric (scale = max|w|/127) of the dequantized
            Q4_0 weights; FC consumes it directly (classic DRQ). NOT Q4_0-exact.       6 ops
Q4_0 d may be negative; i4bw keeps the signed d verbatim so the int4 form is bit-exact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s

import q4_0

T = s.TensorType


def pack_int4(q, order="lo"):
    flat = q.reshape(-1).astype(np.uint8) & 0x0F
    if order == "hi":
        return ((flat[0::2] << 4) | flat[1::2]).astype(np.uint8)
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def effective(fmt, q, d):
    """(q_int, scale, zero_point) actually stored for the int4 blockwise variants."""
    if fmt == "i4bw_pos":
        neg = (d.astype(np.float32) < 0)
        rows, cols = q.shape
        qb = q.reshape(rows, cols // 32, 32).astype(np.int16)
        qb = np.where(neg[:, :, None], np.clip(-qb, -8, 7), qb).astype(np.int8)
        return qb.reshape(rows, cols), np.abs(d), None
    if fmt == "i4bw_zp":
        neg = (d.astype(np.float32) < 0)
        rows, cols = q.shape
        qb = q.reshape(rows, cols // 32, 32).astype(np.int16)
        qb = np.where(neg[:, :, None], -qb - 1, qb).astype(np.int8)
        return qb.reshape(rows, cols), np.abs(d), np.where(neg, -1, 0).astype(np.int8)
    return q, d, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--weights-format", dest="fmt", required=True, choices=("f32", "f16", "i4bw", "i4bw_dq", "i4bw_pos", "i4bw_zp", "i4emu", "i4wo", "i4pc", "i8pc"))
    ap.add_argument("--scale-dtype", default="f16", choices=("f16", "f32"))
    ap.add_argument("--nibble-order", default="lo", choices=("lo", "hi"))
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--part", default="all", choices=("all", "gateup", "down"))
    ap.add_argument("--columns", type=int, default=q4_0.N_FF, help="keep only the first C FFN columns (smoke tests)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    layer = q4_0.load(args.weights)
    M, E, F = args.batch, q4_0.N_EMBD, args.columns
    if F != q4_0.N_FF:
        if F % q4_0.QK:
            ap.error("columns must be a multiple of 32")
        layer = {"gate": (layer["gate"][0][:F], layer["gate"][1][:F]), "up": (layer["up"][0][:F], layer["up"][1][:F]),
                 "down": (layer["down"][0][:, :F], layer["down"][1][:, :F // q4_0.QK])}

    model = s.ModelT()
    model.version = 3
    model.description = f"Qwen3-32B blk.11 FFN Q4_0 weights as {args.fmt}, M={M}, part={args.part}"
    model.buffers = [s.BufferT()]
    g = s.SubGraphT()
    g.name = "main"
    g.tensors, g.operators = [], []
    model.subgraphs = [g]
    model.operatorCodes = []
    codes = {}

    def opcode(code):
        if code not in codes:
            oc = s.OperatorCodeT()
            oc.builtinCode = code
            oc.deprecatedBuiltinCode = code if code < 127 else 127
            oc.version = 1
            codes[code] = len(model.operatorCodes)
            model.operatorCodes.append(oc)
        return codes[code]

    def tensor(name, shape, ttype=T.FLOAT32, data=None, quant=None):
        t = s.TensorT()
        t.name, t.shape, t.type, t.hasRank = name, list(shape), ttype, True
        if data is not None:
            b = s.BufferT()
            b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), dtype=np.uint8)
            t.buffer = len(model.buffers)
            model.buffers.append(b)
        if quant is not None:
            t.quantization = quant
        g.tensors.append(t)
        return len(g.tensors) - 1

    def op(code, inputs, outputs, opt_type=s.BuiltinOptions.NONE, opt=None):
        o = s.OperatorT()
        o.opcodeIndex, o.inputs, o.outputs = opcode(code), inputs, outputs
        o.builtinOptionsType, o.builtinOptions = opt_type, opt
        g.operators.append(o)

    def fc(x, w, out):
        opt = s.FullyConnectedOptionsT()
        opt.asymmetricQuantizeInputs = False
        op(s.BuiltinOperator.FULLY_CONNECTED, [x, w, -1], [out], s.BuiltinOptions.FullyConnectedOptions, opt)

    weight_bytes = 0

    def const(name, arr, ttype):
        return tensor(name, arr.shape, ttype, arr)

    def per_tensor_unit():
        qp = s.QuantizationParametersT()
        qp.scale = np.ones(1, dtype=np.float32)
        qp.zeroPoint = np.zeros(1, dtype=np.int64)
        qp.quantizedDimension = 0
        return qp

    def emulated_fc(x, name, out):
        """EMULATED_SUBCHANNEL: x [M,K] -> [M,N] with INT4 [1,K/32,32,N] + FP32 scales [1,K/32,1,N]."""
        nonlocal weight_bytes
        q, d = layer[name]
        n, k = q.shape
        nb = k // q4_0.QK
        tname = f"blk.11.ffn_{name}.weight"
        qt = np.ascontiguousarray(q.reshape(n, nb, q4_0.QK).transpose(1, 2, 0)).reshape(1, nb, q4_0.QK, n)
        packed = pack_int4(qt, args.nibble_order)
        wt = tensor(tname, (1, nb, q4_0.QK, n), T.INT4, packed, per_tensor_unit())
        sc = np.ascontiguousarray(d.astype(np.float32).T).reshape(1, nb, 1, n)
        st = const(tname + "_scale", sc.astype("<f4"), T.FLOAT32)
        weight_bytes += packed.nbytes + sc.nbytes
        shape1 = const(tname + "_rs1", np.array([M, nb, 1, q4_0.QK], np.int32), T.INT32)
        axes = const(tname + "_axes", np.array([1], np.int32), T.INT32)
        shape2 = const(tname + "_rs2", np.array([M, n], np.int32), T.INT32)
        xr = tensor(tname + "_x4", [M, nb, 1, q4_0.QK])
        bm = tensor(tname + "_bmm", [M, nb, 1, n])
        ml = tensor(tname + "_mul", [M, nb, 1, n])
        sm = tensor(tname + "_sum", [M, 1, 1, n])
        ro = s.ReshapeOptionsT(); ro.newShape = [M, nb, 1, q4_0.QK]
        op(s.BuiltinOperator.RESHAPE, [x, shape1], [xr], s.BuiltinOptions.ReshapeOptions, ro)
        bo = s.BatchMatMulOptionsT(); bo.asymmetricQuantizeInputs = False
        op(s.BuiltinOperator.BATCH_MATMUL, [xr, wt], [bm], s.BuiltinOptions.BatchMatMulOptions, bo)
        op(s.BuiltinOperator.MUL, [bm, st], [ml], s.BuiltinOptions.MulOptions, s.MulOptionsT())
        rd = s.ReducerOptionsT(); rd.keepDims = True
        op(s.BuiltinOperator.SUM, [ml, axes], [sm], s.BuiltinOptions.ReducerOptions, rd)
        ro2 = s.ReshapeOptionsT(); ro2.newShape = [M, n]
        op(s.BuiltinOperator.RESHAPE, [sm, shape2], [out], s.BuiltinOptions.ReshapeOptions, ro2)

    def weight(name):
        nonlocal weight_bytes
        q, d = layer[name]
        rows, cols = q.shape
        tname = f"blk.11.ffn_{name}.weight"
        if args.fmt == "f32":
            w = q4_0.dequantize(q, d).astype("<f4")
            weight_bytes += w.nbytes
            return tensor(tname, w.shape, T.FLOAT32, w)
        if args.fmt == "f16":
            w = q4_0.dequantize(q, d).astype("<f2")
            weight_bytes += w.nbytes
            wt = tensor(tname, w.shape, T.FLOAT16, w)
            deq = tensor(tname + ".dequant", w.shape)
            op(s.BuiltinOperator.DEQUANTIZE, [wt], [deq])
            return deq
        if args.fmt in ("i4bw", "i4bw_dq", "i4bw_pos", "i4bw_zp"):
            q, d, zp = effective(args.fmt, q, d)
            scales = d.astype("<f2") if args.scale_dtype == "f16" else d.astype("<f4")
            sc = tensor(tname + "_scales", scales.shape, T.FLOAT16 if args.scale_dtype == "f16" else T.FLOAT32, scales)
            qp = s.QuantizationParametersT()
            qp.detailsType = s.QuantizationDetails.BlockwiseQuantization
            bw = s.BlockwiseQuantizationT()
            zpt = -1
            if zp is not None:
                zpt = tensor(tname + "_zero_points", zp.shape, T.INT8, zp)
            bw.scales, bw.zeroPoints, bw.blockSize = sc, zpt, q4_0.QK
            qp.details = bw
            qp.quantizedDimension = 0
            packed = pack_int4(q, args.nibble_order)
            weight_bytes += packed.nbytes + scales.nbytes
            wt = tensor(tname, (rows, cols), T.INT4, packed, qp)
            if args.fmt == "i4bw":
                return wt
            deq = tensor(tname + ".dequant", (rows, cols))
            op(s.BuiltinOperator.DEQUANTIZE, [wt], [deq])
            return deq
        if args.fmt == "i4wo":
            nb = cols // q4_0.QK
            packed = pack_int4(q, args.nibble_order)
            wt = tensor(tname, (rows, nb, q4_0.QK), T.INT4, packed, per_tensor_unit())
            sc = d.astype("<f2").reshape(rows, nb, 1)
            sct = const(tname + "_scale", sc, T.FLOAT16)
            scf = tensor(tname + "_scale_f32", (rows, nb, 1))
            op(s.BuiltinOperator.DEQUANTIZE, [sct], [scf])
            wq = tensor(tname + ".dequant", (rows, nb, q4_0.QK))
            op(s.BuiltinOperator.DEQUANTIZE, [wt], [wq])
            wm = tensor(tname + ".scaled", (rows, nb, q4_0.QK))
            op(s.BuiltinOperator.MUL, [wq, scf], [wm], s.BuiltinOptions.MulOptions, s.MulOptionsT())
            shp = const(tname + "_shape", np.array([rows, cols], np.int32), T.INT32)
            wf = tensor(tname + ".float", (rows, cols))
            ro = s.ReshapeOptionsT(); ro.newShape = [rows, cols]
            op(s.BuiltinOperator.RESHAPE, [wm, shp], [wf], s.BuiltinOptions.ReshapeOptions, ro)
            weight_bytes += packed.nbytes + sc.nbytes
            return wf
        if args.fmt == "i4pc":
            w = q4_0.dequantize(q, d)
            scale = np.maximum(np.abs(w).max(axis=1), 1e-30) / 7.0
            qi = np.clip(np.rint(w / scale[:, None]), -8, 7).astype(np.int8)
            qp = s.QuantizationParametersT()
            qp.scale = scale.astype(np.float32)
            qp.zeroPoint = np.zeros(rows, dtype=np.int64)
            qp.quantizedDimension = 0
            packed = pack_int4(qi)
            weight_bytes += packed.nbytes
            return tensor(tname, (rows, cols), T.INT4, packed, qp)
        if args.fmt == "i8pc":
            w = q4_0.dequantize(q, d)
            scale = np.maximum(np.abs(w).max(axis=1), 1e-30) / 127.0
            qi = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
            qp = s.QuantizationParametersT()
            qp.scale = scale.astype(np.float32)
            qp.zeroPoint = np.zeros(rows, dtype=np.int64)
            qp.quantizedDimension = 0
            weight_bytes += qi.nbytes
            return tensor(tname, (rows, cols), T.INT8, qi, qp)
        raise AssertionError

    emu = args.fmt == "i4emu"

    def proj(x, name, out):
        if emu:
            emulated_fc(x, name, out)
        else:
            fc(x, weight(name), out)

    if args.part in ("all", "gateup"):
        x = tensor("input", [M, E])
        gate, up, sig, silu = (tensor(n, [M, F]) for n in ("gate", "up", "sigmoid", "silu"))
        proj(x, "gate", gate)
        proj(x, "up", up)
        op(s.BuiltinOperator.LOGISTIC, [gate], [sig])
        op(s.BuiltinOperator.MUL, [gate, sig], [silu], s.BuiltinOptions.MulOptions, s.MulOptionsT())
        if args.part == "all":
            prod = tensor("product", [M, F])
            op(s.BuiltinOperator.MUL, [silu, up], [prod], s.BuiltinOptions.MulOptions, s.MulOptionsT())
            out = tensor("output", [M, E])
            proj(prod, "down", out)
        else:
            out = tensor("output", [M, F])
            op(s.BuiltinOperator.MUL, [silu, up], [out], s.BuiltinOptions.MulOptions, s.MulOptionsT())
    else:
        x = tensor("input", [M, F])
        out = tensor("output", [M, E])
        proj(x, "down", out)
    g.inputs, g.outputs = [x], [out]
    sig_def = s.SignatureDefT()
    sig_def.signatureKey = "serving_default"
    sig_def.inputs, sig_def.outputs = [], []
    for name, idx, lst in (("input", x, sig_def.inputs), ("output", out, sig_def.outputs)):
        tm = s.TensorMapT()
        tm.name, tm.tensorIndex = name, idx
        lst.append(tm)
    model.signatureDefs = [sig_def]
    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    dest = args.output / "ffn.tflite"
    dest.write_bytes(b.Output())
    with dest.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    rec = dict(fmt=args.fmt, scale_dtype=args.scale_dtype, nibble_order=args.nibble_order, batch=M, part=args.part, columns=F, ops=len(g.operators), tflite_bytes=dest.stat().st_size,
               weight_payload_bytes=weight_bytes, model_sha256=sha,
               weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
               exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "MODEL.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
