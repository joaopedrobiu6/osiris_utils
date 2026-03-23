#!/bin/bash -l
#SBATCH --job-name=create_db
#SBATCH --time=08:00:00
#SBATCH --account=epor47
#SBATCH --qos=gp_ehpc
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # matches --workers below; honest SLURM accounting
#SBATCH --exclusive               # full node: no memory pressure from other users
#SBATCH --output=logs/create_db_%A_%a.out
#SBATCH --error=logs/create_db_%A_%a.err

# ── Modules ──────────────────────────────────────────────────────────────────
module purge
module load hdf5 python

# ── Thread / library settings ─────────────────────────────────────────────────
# MKL: 2 threads per Python worker → 16 workers × 2 = 32 physical cores used.
# Keeps BLAS fast for numpy ops inside each frame without oversubscribing.
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2           # OpenMP (e.g. scipy, some numpy builds)
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# HDF5 per-file chunk cache (respected by OsirisGridFile._open_file_hdf5).
# 256 MB gives ~6× better sequential read throughput vs. the default 1 MB.
# Safe: cache is per open file handle; peak overhead = workers × 256 MB = 4 GB
# for 16 workers — well within the ~256 GB available on an exclusive MN5 node.
export OSIRIS_HDF5_CACHE_MB=256

# ── Paths ─────────────────────────────────────────────────────────────────────
SIM_IN="/gpfs/scratch/epor47/joaobiu/simulations/elec_ion_015c_seed_11/shock_2D_elec_ion.in"
OUTDIR="/gpfs/scratch/epor47/joaobiu/db_slices_seed0_800_4800_for_boosts"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

T_START=800
T_END=4800
CHUNK=400

# ── Per-task slice ────────────────────────────────────────────────────────────
t0=$((T_START + SLURM_ARRAY_TASK_ID * CHUNK))
t1=$((t0 + CHUNK))
if [ "$t1" -gt "$T_END" ]; then t1=$T_END; fi
if [ "$t0" -ge "$T_END" ]; then
    echo "Nothing to do for task ${SLURM_ARRAY_TASK_ID} (t0=$t0 >= T_END=$T_END)."
    exit 0
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR" logs

# Lustre striping for the output directory (run once; harmless if already set).
# Stripes writes across 8 OSTs → ~8× write bandwidth for large .npy files.
lfs setstripe -c 8 "$OUTDIR" 2>/dev/null || true

echo "======================================================="
echo "Host         : $(hostname)"
echo "Array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Slice        : [$t0, $t1)"
echo "Output dir   : $OUTDIR"
echo "Workers      : 16  (MKL_NUM_THREADS=2 → 32 cores)"
echo "HDF5 cache   : ${OSIRIS_HDF5_CACHE_MB} MB/file"
echo "======================================================="

# ── Run ───────────────────────────────────────────────────────────────────────
srun python3 "$SCRIPT_DIR/create_databases.py" \
    --sim-in  "$SIM_IN"   \
    --outdir  "$OUTDIR"   \
    --t0      "$t0"       \
    --t1      "$t1"       \
    --workers 16
