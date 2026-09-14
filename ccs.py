#!/usr/bin/env python3
"""Run with: python ccs.py molecule.xyz --output results/molecule"""
from pathlib import Path
import os
import sys

if __name__ == '__main__':
    # Serial BLAS avoids hidden thread multiplication; --threads controls EHSS.
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ.setdefault(name,'1')
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    try:
        from support_exchange_transport.ccs_cli import main
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: python -m pip install -r requirements-ccs.txt",file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
