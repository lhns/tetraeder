"""Analysis of the s3e traces: per-rep samples_ns per run and the per-step per-op sums of each trace (call 1 direct,
call 2 capture, call 3+ replay), then everything s3d_analyze prints.

    python3 scripts/s3e_analyze.py <session dir>
"""

import sys

import s3d_analyze

if __name__ == "__main__":
    s3d_analyze.show_samples(sys.argv[1])
    s3d_analyze.show_steps(sys.argv[1])
    s3d_analyze.main(sys.argv[1])
