"""Stdlib-only GGUF header index: prints tensor name, ggml type, shape, absolute byte offset and size.

Used on the desktop (no gguf/numpy needed) to locate one layer's FFN tensors so that only
those byte ranges are copied. Usage: python3 gguf_index.py FILE [name-substring]
"""
import json
import struct
import sys

# ggml type -> (block elements, block bytes)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
         8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        def rstr():
            (n,) = rd("Q")
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if t in scalar:
                return rd(scalar[t])[0]
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = rd("I")
                (n,) = rd("Q")
                vals = [rval(et) for _ in range(n)]
                return vals if n <= 16 else f"<array {n}>"
            raise ValueError(t)

        magic = f.read(4)
        assert magic == b"GGUF", magic
        (version,) = rd("I")
        (n_tensors,) = rd("Q")
        (n_kv,) = rd("Q")
        kv = {}
        alignment = 32
        for _ in range(n_kv):
            key = rstr()
            (t,) = rd("I")
            kv[key] = rval(t)
            if key == "general.alignment":
                alignment = kv[key]
        infos = []
        for _ in range(n_tensors):
            name = rstr()
            (nd,) = rd("I")
            dims = list(rd("Q" * nd))
            (ggml_type,) = rd("I")
            (offset,) = rd("Q")
            infos.append((name, dims, ggml_type, offset))
        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
    out = {"version": version, "n_tensors": n_tensors, "data_start": data_start,
           "kv": {k: v for k, v in kv.items() if not isinstance(v, str) or len(v) < 200}, "tensors": []}
    for name, dims, ggml_type, offset in infos:
        if want and want not in name:
            continue
        tname, be, bb = TYPES.get(ggml_type, (str(ggml_type), 0, 0))
        n = 1
        for d in dims:
            n *= d
        size = n // be * bb if be else None
        out["tensors"].append({"name": name, "dims_ggml": dims, "type": tname, "type_id": ggml_type,
                               "abs_offset": data_start + offset, "bytes": size})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
