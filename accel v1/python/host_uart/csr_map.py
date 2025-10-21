
"""
csr_map.py - Accel v1 CSR register map (NVIDIA/industry style)
------------------------------------------------------------
* Little-endian, 32-bit aligned, byte-addressed register map
* Explicit offsets, bitfields, and type metadata for auto-docs and C header generation
* Robust helpers for packing/unpacking, framing, and config serialization
"""

from dataclasses import dataclass, fields
import struct
from typing import Any, Dict, List, Tuple

LE = "<"  # little-endian

# Register byte offsets (32-bit aligned)
CTRL         = 0x00  #: Control register
DIMS_M       = 0x04  #: Matrix M dimension
DIMS_N       = 0x08  #: Matrix N dimension
DIMS_K       = 0x0C  #: Matrix K dimension
TILES_Tm     = 0x10  #: Tile size Tm
TILES_Tn     = 0x14  #: Tile size Tn
TILES_Tk     = 0x18  #: Tile size Tk
INDEX_m      = 0x1C  #: m index
INDEX_n      = 0x20  #: n index
INDEX_k      = 0x24  #: k index
BUFF         = 0x28  #: Buffer control
SCALE_Sa     = 0x2C  #: Activation scale (float32)
SCALE_Sw     = 0x30  #: Weight scale (float32)
UART_len_max = 0x34  #: UART max packet length
UART_crc_en  = 0x38  #: UART CRC enable
STATUS       = 0x3C  #: Status register

# CTRL bits
CTRL_START = 1 << 0  #: Start pulse (W1P)
CTRL_ABORT = 1 << 1  #: Abort pulse (W1P)
CTRL_IRQEN = 1 << 2  #: Interrupt enable (RW)

# STATUS bits
STS_BUSY        = 1 << 0  #: Accelerator busy (RO)
STS_DONE_TILE   = 1 << 1  #: Tile done (R/W1C)
STS_ERR_CRC     = 1 << 8  #: CRC error (R/W1C)
STS_ERR_ILLEGAL = 1 << 9  #: Illegal op error (R/W1C)

# BUFF bits
WR_A = 1 << 0  #: Write bank A
WR_B = 1 << 1  #: Write bank B
RD_A = 1 << 8  #: Read bank A
RD_B = 1 << 9  #: Read bank B

# Field/type metadata for auto-docs and validation
CSR_LAYOUT = [
    (CTRL,         "CTRL",         "u32", "Control register"),
    (DIMS_M,       "DIMS_M",       "u32", "Matrix M dimension"),
    (DIMS_N,       "DIMS_N",       "u32", "Matrix N dimension"),
    (DIMS_K,       "DIMS_K",       "u32", "Matrix K dimension"),
    (TILES_Tm,     "TILES_Tm",     "u32", "Tile size Tm"),
    (TILES_Tn,     "TILES_Tn",     "u32", "Tile size Tn"),
    (TILES_Tk,     "TILES_Tk",     "u32", "Tile size Tk"),
    (INDEX_m,      "INDEX_m",      "u32", "m index"),
    (INDEX_n,      "INDEX_n",      "u32", "n index"),
    (INDEX_k,      "INDEX_k",      "u32", "k index"),
    (BUFF,         "BUFF",         "u32", "Buffer control bits"),
    (SCALE_Sa,     "SCALE_Sa",     "f32", "Activation scale"),
    (SCALE_Sw,     "SCALE_Sw",     "f32", "Weight scale"),
    (UART_len_max, "UART_len_max", "u32", "UART max packet length"),
    (UART_crc_en,  "UART_crc_en",  "u32", "UART CRC enable"),
    (STATUS,       "STATUS",       "u32", "Status register"),
]

FIELD_TYPES = {
    "u32": (4, lambda x: struct.pack(LE + "I", x), lambda b: struct.unpack(LE + "I", b)[0]),
    "f32": (4, lambda x: struct.pack(LE + "f", x), lambda b: struct.unpack(LE + "f", b)[0]),
}

def pack_u32(x: int) -> bytes:
    """Pack unsigned 32-bit integer (little-endian)"""
    return struct.pack(LE + "I", x)

def pack_f32(x: float) -> bytes:
    """Pack 32-bit float (little-endian)"""
    return struct.pack(LE + "f", x)

def unpack_u32(b: bytes) -> int:
    """Unpack unsigned 32-bit integer (little-endian)"""
    return struct.unpack(LE + "I", b)[0]

def unpack_f32(b: bytes) -> float:
    """Unpack 32-bit float (little-endian)"""
    return struct.unpack(LE + "f", b)[0]

@dataclass
class Config:
    """Accelerator configuration (matches CSR layout)"""
    M: int
    N: int
    K: int
    Tm: int
    Tn: int
    Tk: int
    m_idx: int = 0
    n_idx: int = 0
    k_idx: int = 0
    Sa: float = 1.0
    Sw: float = 1.0
    wrA: int = 0
    wrB: int = 0

    def to_bytes(self) -> bytes:
        """Serialize config to register image (for dumps or programming)"""
        reg_img = bytearray(STATUS + 4)
        for addr, name, typ, _ in CSR_LAYOUT:
            val = getattr(self, name.split('_')[1].lower(), 0) if name.startswith('DIMS_') or name.startswith('TILES_') or name.startswith('INDEX_') else getattr(self, name.split('_')[0].lower(), 0)
            if name == "SCALE_Sa": val = self.Sa
            if name == "SCALE_Sw": val = self.Sw
            if name == "BUFF": val = (self.wrA & 1)*WR_A | (self.wrB & 1)*WR_B
            if name == "CTRL": val = 0 # Not set here
            if name == "STATUS": val = 0 # Not set here
            sz, pack, _ = FIELD_TYPES[typ]
            reg_img[addr:addr+sz] = pack(val)
        return bytes(reg_img)

    @classmethod
    def from_bytes(cls, b: bytes) -> 'Config':
        """Deserialize config from register image"""
        kwargs = {}
        for addr, name, typ, _ in CSR_LAYOUT:
            sz, _, unpack = FIELD_TYPES[typ]
            if addr+sz > len(b): continue
            val = unpack(b[addr:addr+sz])
            if name == "SCALE_Sa": kwargs['Sa'] = val
            elif name == "SCALE_Sw": kwargs['Sw'] = val
            elif name == "BUFF":
                kwargs['wrA'] = 1 if (val & WR_A) else 0
                kwargs['wrB'] = 1 if (val & WR_B) else 0
            elif name.startswith('DIMS_'):
                kwargs[name.split('_')[1].lower()] = val
            elif name.startswith('TILES_'):
                kwargs[name.split('_')[1].lower()] = val
            elif name.startswith('INDEX_'):
                kwargs[name.split('_')[1].lower() + '_idx'] = val
        return cls(**kwargs)

def to_writes(cfg: Config) -> List[Tuple[int, bytes]]:
    """List of (addr, payload_bytes) tuples to program CSRs."""
    return [
        (DIMS_M,       pack_u32(cfg.M)),
        (DIMS_N,       pack_u32(cfg.N)),
        (DIMS_K,       pack_u32(cfg.K)),
        (TILES_Tm,     pack_u32(cfg.Tm)),
        (TILES_Tn,     pack_u32(cfg.Tn)),
        (TILES_Tk,     pack_u32(cfg.Tk)),
        (INDEX_m,      pack_u32(cfg.m_idx)),
        (INDEX_n,      pack_u32(cfg.n_idx)),
        (INDEX_k,      pack_u32(cfg.k_idx)),
        (BUFF,         pack_u32((cfg.wrA & 1)*WR_A | (cfg.wrB & 1)*WR_B)),
        (SCALE_Sa,     pack_f32(cfg.Sa)),
        (SCALE_Sw,     pack_f32(cfg.Sw)),
    ]

def make_ctrl_start(irq_en: bool) -> bytes:
    """Pack CTRL register for start pulse (optionally enable IRQ)"""
    word = (CTRL_IRQEN if irq_en else 0) | CTRL_START
    return pack_u32(word)

def make_ctrl_abort() -> bytes:
    """Pack CTRL register for abort pulse"""
    return pack_u32(CTRL_ABORT)

# UART framing (no external IO here; just the packers)
SOF = 0xA5
CMD_WRITE = 0x01
CMD_READ  = 0x02

def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16-CCITT (False) for UART packets"""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if (crc & 0x8000):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def frame_write(addr: int, payload: bytes, crc_en: bool=True) -> bytes:
    """Frame a UART write packet with optional CRC"""
    header = struct.pack(LE + "BBIH", SOF, CMD_WRITE, addr, len(payload))
    pkt = header + payload
    if crc_en:
        crc = crc16_ccitt_false(pkt)
        pkt += struct.pack(LE + "H", crc)
    return pkt

def frame_read(addr: int, nbytes: int, crc_en: bool=True) -> bytes:
    """Frame a UART read packet with optional CRC"""
    header = struct.pack(LE + "BBIH", SOF, CMD_READ, addr, 0)
    pkt = header
    if crc_en:
        crc = crc16_ccitt_false(pkt)
        pkt += struct.pack(LE + "H", crc)
    return pkt
