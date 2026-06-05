"""Minimal SPSS .sav (system file) reader.
Supports $FL2 files with bytecode (type-1) compression, which is the standard
PISA distribution format. Returns a pandas DataFrame of numeric/string values.
Reference: PSPP / GNU 'system file format' documentation.
"""
import struct
import numpy as np
import pandas as pd


class SavReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.endian = "<"  # little-endian (typical)
        self.vars = []          # list of dicts for each real variable
        self.var_index = []     # mapping of 8-byte 'elements' -> variable or None (continuation)
        self.compressed = False
        self.bias = 100.0
        self.n_cases = -1
        self.names = []
        self.long_name_map = {}
        self._parse_header()

    def _r(self, n):
        return self.f.read(n)

    def _i32(self):
        return struct.unpack(self.endian + "i", self.f.read(4))[0]

    def _flt(self):
        return struct.unpack(self.endian + "d", self.f.read(8))[0]

    def _parse_header(self):
        rec_type = self._r(4)
        assert rec_type == b"$FL2", "Not an SPSS .sav file"
        self._r(60)              # product name
        layout = self._i32()     # layout code (2 or 3); detect endianness
        if layout not in (2, 3):
            self.endian = ">"
            self.f.seek(64)
            layout = self._i32()
        self.nominal_case_size = self._i32()
        self.compression = self._i32()       # 0 none, 1 bytecode, 2 zsav
        self.weight_index = self._i32()
        self.n_cases = self._i32()
        self.bias = self._flt()
        self._r(9)               # creation date
        self._r(8)               # creation time
        self._r(64)              # file label
        self._r(3)               # padding
        self.compressed = (self.compression == 1)
        assert self.compression in (0, 1), f"Unsupported compression {self.compression}"

        # ---- command/record loop ----
        while True:
            rec_type = self._i32()
            if rec_type == 2:
                self._parse_variable_record()
            elif rec_type == 3:
                self._skip_value_labels()
            elif rec_type == 4:
                # value-label var index record (shouldn't appear standalone here)
                count = self._i32()
                self._r(4 * count)
            elif rec_type == 6:
                # document record
                n = self._i32()
                self._r(80 * n)
            elif rec_type == 7:
                self._parse_extension_record()
            elif rec_type == 999:
                self._r(4)  # filler
                break
            else:
                raise ValueError(f"Unknown record type {rec_type} at {self.f.tell()}")

        # apply long variable names if present
        if self.long_name_map:
            for v in self.vars:
                short = v["name"]
                if short in self.long_name_map:
                    v["short"] = short
                    v["name"] = self.long_name_map[short]
        self.names = [v["name"] for v in self.vars]
        self.data_start = self.f.tell()

    def _parse_variable_record(self):
        type_code = self._i32()       # 0 numeric, >0 string width, -1 continuation
        has_label = self._i32()
        n_missing = self._i32()
        self._r(4)                    # print format
        self._r(4)                    # write format
        name = self._r(8).decode("latin-1").strip()
        if has_label:
            label_len = self._i32()
            self._r(label_len)
            # padded to multiple of 4
            pad = (4 - (label_len % 4)) % 4
            self._r(pad)
        missing_vals = []
        miss_range = None
        if n_missing != 0:
            raw = [self._flt() for _ in range(abs(n_missing))]
            if n_missing < 0:
                # range: first two are lo/hi, optional third discrete
                miss_range = (raw[0], raw[1])
                if len(raw) == 3:
                    missing_vals = [raw[2]]
            else:
                missing_vals = raw

        if type_code == -1:
            # continuation of a long string variable
            self.var_index.append(None)
            return
        var = {
            "name": name,
            "type": type_code,        # 0 numeric, >0 string bytes
            "width": type_code,
            "n_segments": 1,
            "missing_vals": missing_vals,
            "missing_range": miss_range,
        }
        self.vars.append(var)
        self.var_index.append(var)

    def _skip_value_labels(self):
        label_count = self._i32()
        for _ in range(label_count):
            self._r(8)               # value (double)
            n = self.f.read(1)[0]
            total = n + 1
            pad = (8 - (total % 8)) % 8
            self._r(n + pad)
        # paired type-4 record
        rec_type = self._i32()
        assert rec_type == 4
        count = self._i32()
        self._r(4 * count)

    def _parse_extension_record(self):
        subtype = self._i32()
        size = self._i32()
        count = self._i32()
        nbytes = size * count
        data = self._r(nbytes)
        if subtype == 13:
            # long variable names: "SHORT=LongName\tSHORT2=LongName2..."
            try:
                txt = data.decode("latin-1")
            except Exception:
                txt = ""
            mapping = {}
            for pair in txt.split("\t"):
                if "=" in pair:
                    short, long = pair.split("=", 1)
                    mapping[short.strip()] = long.strip()
            self.long_name_map = mapping
        # other subtypes ignored

    # -------------------------------------------------------------------------
    def read(self):
        self.f.seek(self.data_start)
        n_elements = len(self.var_index)
        # Build per-element decode plan: numeric (8 bytes) vs string segment
        # Each element is one 8-byte slot.
        elem_is_string = []
        for v in self.var_index:
            if v is None:
                elem_is_string.append(True)   # continuation -> string bytes
            elif v["type"] > 0:
                elem_is_string.append(True)
            else:
                elem_is_string.append(False)

        # group elements back into variables for assembly
        # numeric var = 1 element; string var = ceil(width/8) elements incl continuations
        var_layout = []   # (var, n_elements, is_string)
        i = 0
        vi = self.var_index
        while i < len(vi):
            v = vi[i]
            if v is None:
                i += 1
                continue
            if v["type"] == 0:
                var_layout.append((v, 1, False))
                i += 1
            else:
                # consume this + following None continuations
                width = v["type"]
                n_seg = 1
                j = i + 1
                while j < len(vi) and vi[j] is None:
                    n_seg += 1
                    j += 1
                var_layout.append((v, n_seg, True))
                i = j

        sysmis = struct.unpack(self.endian + "d", b"\xff\xff\xff\xff\xff\xff\xef\xff" if self.endian == "<" else b"\xff\xef\xff\xff\xff\xff\xff\xff")[0]

        columns = {v["name"]: [] for v, _, _ in var_layout}

        def read_elements(n):
            """Return list of n raw 8-byte elements, decompressing if needed."""
            out = []
            while len(out) < n:
                out.append(self._next_element())
            return out

        if self.compressed:
            self._cmd = b""
            self._cmd_pos = 0
            self._pending = []

        for case in range(self.n_cases):
            for v, n_seg, is_str in var_layout:
                elems = read_elements(n_seg)
                if is_str:
                    raw = b"".join(e if isinstance(e, bytes) else b"        " for e in elems)
                    try:
                        s = raw.decode("latin-1").rstrip()
                    except Exception:
                        s = ""
                    columns[v["name"]].append(s)
                else:
                    e = elems[0]
                    if e is None:
                        columns[v["name"]].append(np.nan)
                    else:
                        val = struct.unpack(self.endian + "d", e)[0]
                        if val == sysmis or val != val:
                            columns[v["name"]].append(np.nan)
                        else:
                            mv = v.get("missing_vals", [])
                            mr = v.get("missing_range", None)
                            is_miss = False
                            if mv and any(abs(val - x) < 1e-9 for x in mv):
                                is_miss = True
                            if mr is not None and mr[0] <= val <= mr[1]:
                                is_miss = True
                            columns[v["name"]].append(np.nan if is_miss else val)

        df = pd.DataFrame(columns)
        return df

    # element reader for compressed/uncompressed --------------------------------
    def _next_element(self):
        if not self.compressed:
            b = self._r(8)
            return b
        # bytecode compression
        while True:
            if self._cmd_pos >= len(self._cmd):
                self._cmd = self._r(8)
                self._cmd_pos = 0
                if len(self._cmd) < 8:
                    raise EOFError
            code = self._cmd[self._cmd_pos]
            self._cmd_pos += 1
            if code == 0:
                continue  # padding / ignore
            elif 1 <= code <= 251:
                return struct.pack(self.endian + "d", code - self.bias)
            elif code == 252:
                raise EOFError("end of file marker")
            elif code == 253:
                return self._r(8)             # raw uncompressed value follows
            elif code == 254:
                return b"        "            # all-blanks string (8 spaces)
            elif code == 255:
                return None                   # system missing


def read_sav(path, usecols=None):
    r = SavReader(path)
    df = r.read()
    if usecols is not None:
        keep = [c for c in usecols if c in df.columns]
        df = df[keep]
    return df


if __name__ == "__main__":
    import sys
    df = read_sav(sys.argv[1])
    print(df.shape)
    print(df.head())
