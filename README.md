# MechanoChat

# Installation

MechanoChat requires Python >= 3.9 and an NVIDIA GPU.

We recommend using Conda or Mamba for environment management. To install Mamba, see https://github.com/conda-forge/miniforge.

**Step 1: Create and activate a virtual environment**
```bash
mamba create -n mc_39 python=3.9
mamba activate mc_39
```

**Step 2: Install the CUDA toolkit**
```bash
mamba install cudatoolkit -c conda-forge   # installs CUDA 11.8 by default
```

**Step 3: Install MechanoChat (with the matching GPU bindings)**

Choose the extra according to your **CUDA toolkit major version** (step 2 installs 11.x):
```bash
pip install -e ".[cuda11]"   # for CUDA 11.x toolkit (default)
# pip install -e ".[cuda12]" # for CUDA 12.x toolkit
```

The `[cuda11]` / `[cuda12]` extra pulls in NVIDIA's official `cuda-python` bindings.
This is **required on recent NVIDIA drivers (CUDA 13 / driver 580+)**, where numba 0.60's
built-in ctypes driver bindings cause a segfault when initializing the CUDA context.

> If you are on an older driver and prefer the legacy behavior, `pip install -e .`
> (without the extra) still works, but GPU simulation will crash on CUDA 13+ drivers.

You can check your driver / CUDA version with `nvidia-smi`.

**Step 4: Enable the NVIDIA bindings via an environment variable (required on CUDA 13+)**

Installing `cuda-python` is not enough — numba only uses it when
`NUMBA_CUDA_USE_NVIDIA_BINDING=1` is set **before numba is first imported**.
This cannot be done reliably at install time (a `pip install` cannot set a runtime
environment variable for your future shell/Jupyter sessions), so configure it at the
**environment level**. Pick the one that matches how you run MechanoChat:

```bash
# Option A — conda activation hook (covers terminal use; applies on every `conda activate`)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export NUMBA_CUDA_USE_NVIDIA_BINDING=1' \
    > "$CONDA_PREFIX/etc/conda/activate.d/numba_cuda_binding.sh"
# re-activate the env for it to take effect
mamba activate mc_39
```

For Jupyter / VSCode notebooks, also set it in the kernel so it applies even when the
kernel is launched without shell activation. Add an `"env"` block to the kernel's
`kernel.json` (find it with terminal command `jupyter kernelspec list`):

```json
{
  "argv": ["...python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "env": { "NUMBA_CUDA_USE_NVIDIA_BINDING": "1" }
}
```

> ⚠️ Why an env var and not code: numba freezes this switch at `import numba` time, and
> numba is often imported indirectly (e.g. `import scanpy` pulls it in via umap/pynndescent)
> *before* `import mechanochat` runs. So setting the variable from inside the package is too
> late. See `docs/cuda13_migration_report.md` §3.1.

# Usage

> ⚠️ **Before running any GPU simulation (`sem.sim_gpu_opt_test(...)`), make sure
> `NUMBA_CUDA_USE_NVIDIA_BINDING=1` is set** (see Installation, Step 4). Without it, on
> CUDA 13 / driver 580+ the kernel crashes (segfault) on the first CUDA call, with no
> Python traceback.
>
> If you cannot configure the environment level and must set it in a notebook, do it in
> the **very first cell, before any other import** (including `scanpy`), then **restart
> the kernel**:
>
> ```python
> import os
> os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"
> # — must run before importing scanpy / mechanochat / numba —
> ```
>
> Quick self-check:
> ```python
> import numba.cuda as c
> assert c.cudadrv.driver.USE_NV_BINDING == 1, "env var not in effect — restart the kernel"
> ```

# Documentation

See detailed documentation at 