#!/usr/bin/env python3
"""Canonical entry point for flow-initialized checkerboard MH.

The implementation remains in the historically named module because that
module is also the reproducible July-24 record.  Its lattice-size arguments
are generic; it is not restricted to L8 -> L16.
"""

from run_lam1p0_mit_coordinate_mh_L8to16 import main


if __name__ == "__main__":
    raise SystemExit(main())
