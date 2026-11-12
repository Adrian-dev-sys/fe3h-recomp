"""The code patches in an Atmosphere-format cheat file (cheats.txt in the project folder, never committed).

A Switch cheat that writes an instruction into the game's code (opcode 0, 4 bytes, the main module) can't patch a
recompiled game at run time: the instruction is C by then. gen.py compiles each such instruction in as a variant
instead, beside the original, and the runtime's cheat VM (runtime/hle/cheatvm.c) picks it while the cheat is on.

code_variants(path, in_text) -> {offset from main's base: [instruction words]}
"""
import os
import re


def opcodes(words):
    """(type, words) per opcode of an Atmosphere cheat VM program; stops at a type it doesn't know the length of."""
    i = 0
    while i < len(words):
        w = words[i]
        t = w >> 28
        if t in (0, 1):
            n = 4 if ((w >> 24) & 0xF) == 8 else 3
        elif t == 2:
            n = 1
        elif t == 3:
            n = 2 if ((w >> 24) & 0xF) == 0 else 1  # (300R0000 VVVVVVVV begins a loop, 310R0000 ends it)
        elif t == 4:
            n = 3
        elif t == 5:
            n = 2
        elif t == 6:
            n = 3
        elif t == 7:
            n = 2
        elif t == 8:
            n = 1
        else:
            return
        yield t, words[i:i + n]
        i += n


def parse(path):
    """[(name, [words])] for each cheat in the file (master codes {name} included)."""
    cheats, cur = [], None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^[\[{](.*)[\]}]$', line)
            if m:
                cur = (m.group(1).strip(), [])
                cheats.append(cur)
                continue
            if cur is None or line.startswith('#'):  # (# lines: the cheat's description)
                continue
            for tok in line.split():
                if re.fullmatch(r'[0-9A-Fa-f]{8}', tok):
                    cur[1].append(int(tok, 16))
    return cheats


def _bid_match(name, bid):
    """A patch file's build id (hex, trailing zero bytes may be left out) names this module."""
    name = name.lower()
    return len(name) >= 16 and all(c in '0123456789abcdef' for c in name) and bid.ljust(64, '0').startswith(name) \
        and set(bid.ljust(64, '0')[len(name):]) <= {'0'}


def _pchtxt(path, bid):
    """[(image offset, bytes)] of a .pchtxt's enabled patches for this build."""
    out, ours, on, shift = [], False, False, 0
    for line in open(path, encoding='utf-8', errors='replace'):
        s = line.strip()
        if s.startswith('@nsobid-'):
            ours = _bid_match(re.match(r'[0-9A-Fa-f]*', s[8:]).group(0), bid)
        elif s.startswith('@flag offset_shift'):
            shift = int(s.split()[-1], 0)
        elif s.startswith('@enabled'):
            on = True
        elif s.startswith('@disabled') or s.startswith('@stop'):
            on = False
        elif on and ours:
            m = re.match(r'([0-9A-Fa-f]+)\s+(?:"(.*)"|([0-9A-Fa-f]+))', s)
            if m:
                data = m.group(2).encode() + b'\0' if m.group(2) is not None else bytes.fromhex(m.group(3)[:len(m.group(3)) // 2 * 2])
                out.append((int(m.group(1), 16) + shift - 0x100, data))
    return out


def _ips(path):
    raw = open(path, 'rb').read()
    ips32 = raw.startswith(b'IPS32')
    if not ips32 and not raw.startswith(b'PATCH'):
        return []
    out, i, ol = [], 5, 4 if ips32 else 3
    while i + ol <= len(raw) and raw[i:i + ol] != (b'EEOF' if ips32 else b'EOF'):
        off = int.from_bytes(raw[i:i + ol], 'big')
        n = int.from_bytes(raw[i + ol:i + ol + 2], 'big')
        i += ol + 2
        if n == 0:
            n, v = int.from_bytes(raw[i:i + 2], 'big'), raw[i + 2]
            data, i = bytes([v]) * n, i + 3
        else:
            data, i = raw[i:i + n], i + n
        out.append((off - 0x100, data))
    return out


def mod_variants(mods_dir, bid, in_text, word):
    """{offset: [words]} of the code patches in mods/<name>/exefs (.pchtxt / .ips) for build `bid` (runtime/mods.c)."""
    out = {}
    if not mods_dir or not os.path.isdir(mods_dir):
        return out
    for mod in sorted(os.listdir(mods_dir)):
        ex = os.path.join(mods_dir, mod, 'exefs')
        if not os.path.isdir(ex):
            continue
        for fn in sorted(os.listdir(ex)):
            stem, ext = os.path.splitext(fn)
            p = os.path.join(ex, fn)
            if ext.lower() == '.pchtxt':
                patches = _pchtxt(p, bid)
            elif ext.lower() == '.ips' and _bid_match(stem, bid):
                patches = _ips(p)
            else:
                continue
            for off, data in patches:
                if off < 0:
                    continue
                for w in range(off & ~3, off + len(data), 4):
                    if not in_text(w):
                        continue
                    b = bytearray(word(w).to_bytes(4, 'little'))
                    for k in range(4):
                        if off <= w + k < off + len(data):
                            b[k] = data[w + k - off]
                    v = int.from_bytes(b, 'little')
                    if v == word(w):
                        continue
                    vs = out.setdefault(w, [])
                    if v not in vs:
                        vs.append(v)
    return out


def code_variants(path, in_text):
    out = {}
    if not path or not os.path.exists(path):
        return out
    for name, words in parse(path):
        for t, op in opcodes(words):
            w = op[0]
            width, mem, reg = (w >> 24) & 0xF, (w >> 20) & 0xF, (w >> 16) & 0xF
            if t != 0 or width != 4 or mem != 0 or reg != 0:  # (a register offset: not known before it runs)
                continue
            off = ((w & 0xFF) << 32) | op[1]
            if not in_text(off):
                continue
            vs = out.setdefault(off, [])
            if op[2] not in vs:
                vs.append(op[2])
    return out
