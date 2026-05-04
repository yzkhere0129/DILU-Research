 **Date**: 2026-05-04
  **Case**: spot_melt_150W (50×200×50 = 500K cells, OF v2506)
  **Hardware**: Xeon Gold 5120, 28 phys cores × 2 SMT, mpirun -np 32 --oversubscribe
  **Run**: 78 timesteps, 234 pd solves total
## pd_corr0 wall (median over 78 steps)

  - min=6.68 ms, **median=49.57 ms**, max=210.28 ms
  - (mean over all 234 pd solves: 17.83 ms)
