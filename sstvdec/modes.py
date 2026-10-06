"""
SSTV mode timing tables.

Each mode is described by its repeating "unit" (the stretch between two
successive sync pulses), measured in milliseconds from the START of that
unit's sync pulse. A unit carries `rows` image rows. Components list which
channel (Y, Cb, Cr for YCbCr modes; R, G, B for RGB modes) is scanned when,
and which of the unit's rows it belongs to.

Timings: Dayton paper (JL Barber, "Proposal for SSTV mode specifications"),
cross-checked against pysstv's encoders for Robot36, PD and Martin.
Scottie uses the published 428.22 ms line. EXPERIMENTAL: pysstv's Scottie
puts a gap on both sides of every channel (non-standard), so it could not
be validated against an independent encoder. The ISS has not used Scottie.

Robot36 is special: one chroma channel per line, alternating Cr/Cb, chosen by
the separator tone (1500 Hz -> Cr, 2300 Hz -> Cb); each chroma line is
shared by two rows. Handled with channel "C" + parity logic in the decoder.
"""

from dataclasses import dataclass, field

FREQ_BLACK, FREQ_WHITE, FREQ_SYNC = 1500.0, 2300.0, 1200.0


@dataclass(frozen=True)
class Comp:
    chan: str          # Y Cb Cr R G B, or C (Robot36 alternating chroma)
    row: int           # which row of the unit (0..rows-1); -1 = all rows
    t0: float          # ms from unit sync start
    dur: float         # ms
    px: int            # pixels across


@dataclass(frozen=True)
class Mode:
    name: str
    vis: int
    width: int
    height: int
    color: str         # "YCbCr" or "RGB"
    sync_ms: float
    unit_ms: float
    rows: int          # image rows per unit
    comps: tuple = field(default_factory=tuple)
    sep_ms: float = 0.0   # Robot36: separator tone start (parity detection)

    @property
    def units(self):
        return self.height // self.rows

    @property
    def duration_s(self):
        return self.units * self.unit_ms / 1000.0


def _robot36():
    # sync 9 | porch 3 | Y 88 | sep 4.5 | porch 1.5 | C 44   = 150 ms
    return Mode("Robot36", 8, 320, 240, "YCbCr", 9.0, 150.0, 1,
                (Comp("Y", 0, 12.0, 88.0, 320), Comp("C", 0, 106.0, 44.0, 160)),
                sep_ms=100.0)


def _pd(name, vis, w, h, px_ms):
    # sync 20 | porch 2.08 | Y(even) | Cr | Cb | Y(odd), each w*px_ms
    s = w * px_ms
    t = 22.08
    return Mode(name, vis, w, h, "YCbCr", 20.0, 22.08 + 4 * s, 2,
                (Comp("Y", 0, t, s, w), Comp("Cr", -1, t + s, s, w),
                 Comp("Cb", -1, t + 2 * s, s, w), Comp("Y", 1, t + 3 * s, s, w)))


def _martin(name, vis, w, scan):
    # sync 4.862 | gap .572 | G | gap | B | gap | R | gap
    g = 0.572
    t = 4.862 + g
    return Mode(name, vis, w, 256, "RGB", 4.862, 4.862 + 4 * g + 3 * scan, 1,
                (Comp("G", 0, t, scan, w), Comp("B", 0, t + scan + g, scan, w),
                 Comp("R", 0, t + 2 * (scan + g), scan, w)))


def _scottie(name, vis, w, scan):
    # ... gap | G | gap | B | sync 9 | gap | R   (sync sits mid-line)
    # relative to this row's sync: R after it, G and B of this row before it
    g = 1.5
    unit = 9.0 + 3 * g + 3 * scan
    return Mode(name, vis, w, 256, "RGB", 9.0, unit, 1,
                (Comp("G", 0, -(2 * scan + g), scan, w),
                 Comp("B", 0, -scan, scan, w),
                 Comp("R", 0, 9.0 + g, scan, w)))


MODES = {m.name: m for m in (
    _robot36(),
    _pd("PD50", 93, 320, 256, 0.286),
    _pd("PD90", 99, 320, 256, 0.532),
    _pd("PD120", 95, 640, 496, 0.190),
    _pd("PD160", 98, 512, 400, 0.382),
    _pd("PD180", 96, 640, 496, 0.286),
    _pd("PD240", 97, 640, 496, 0.382),
    _pd("PD290", 94, 800, 616, 0.286),
    _martin("Martin1", 44, 320, 146.432),
    _martin("Martin2", 40, 160, 73.216),
    _scottie("Scottie1", 60, 320, 138.240),
    _scottie("Scottie2", 56, 160, 88.064),
    _scottie("ScottieDX", 76, 320, 345.600),
)}

BY_VIS = {m.vis: m for m in MODES.values()}

# modes ARISS / ISS events have actually used - searched first
ISS_COMMON = ("PD120", "PD180", "Robot36", "PD90", "PD160", "PD240", "PD50",
              "PD290", "Martin1", "Martin2")
