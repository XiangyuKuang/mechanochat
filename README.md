# MechanoChat

**MechanoChat** infers and visualizes **mechanical cell–cell communication** from spatial transcriptomics data.

![MechanoChat overview](https://raw.githubusercontent.com/XiangyuKuang/mechanochat/main/docs/images/overview.png)

MechanoChat is built on the subcellular element method : each cell is represented by a collection of subcellular elements interacting through Lennard-Jones potentials, with per-cell adhesion and stiffness derived from gene expression. Relaxing this system reconstructs cell shapes and cell–cell contacts from spot/cell coordinates, and generates the forces acting between neighboring cells. These  mechanical quantities are then coupled with expression of mechanosensitive genes (ion channels, transcription factors, ligand–receptor) to compute signaling between every pair of contacting cells.

- **Simulate**: GPU-accelerated (`numba.cuda`) 2D/3D SEM dynamics (`mechanochat.SEM2`, `mechanochat.SEM3`).
- **MechanoChatDB**: prior knowledge of mechanotransduction related genes (`mechanochat.preprocessing`).
- **Infer**: cell–cell contacts, forces, mechanical cell-cell communication, and crosstalk (`mechanochat.tools`).
- **Visualize**: reconstructed cell shapes (alpha-shapes), spatial distributions of mechanical cell-cell communication(`mechanochat.plotting`).

Everything is written into the AnnData object (`.obsp` / `.obsm` / `.uns`), so MechanoChat can be integrated into a standard Scanpy workflow.

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

You can check your driver / CUDA version with `nvidia-smi`. Choose the extra according to your **CUDA toolkit major version** (step 2 installs 11.x):
```bash
pip install "mechanochat[cuda11]"     # for CUDA 11.x toolkit (default)
# pip install "mechanochat[cuda12]"   # for CUDA 12.x toolkit
```

**Do not omit the extra.** It pulls in NVIDIA's official `cuda-python` bindings, which numba
needs on recent NVIDIA drivers (CUDA 13 / driver 580+) — without them, numba 0.60's built-in
ctypes driver bindings cause a segfault when initializing the CUDA context. `cuda-python` is
shipped as an extra rather than a regular dependency only because its major version has to match
the CUDA version on your machine, which cannot be decided at packaging time.

To install the development version from source:
```bash
git clone https://github.com/XiangyuKuang/mechanochat.git
cd mechanochat
pip install -e ".[cuda11]"
```

# Usage

> ⚠️ **Before running simulation (`sem.sim_gpu(...)`), make sure `NUMBA_CUDA_USE_NVIDIA_BINDING=1` is set** . Without it, on CUDA 13 / driver 580+ the kernel crashes (segfault) on the first CUDA call, with no Python traceback.
>
> Do it in the **very first cell, before any other import** (including `scanpy`):
>
> ```python
> import os
> os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"
> # — must run before importing scanpy / numba —
> import mechanochat as mc
> import scanpy as sc
> ```
>
> Quick self-check:
> ```python
> import numba.cuda as c
> assert c.cudadrv.driver.USE_NV_BINDING == 1, "env var not in effect — restart the kernel"
> ```

<details>
<summary><b>Optional — set <code>NUMBA_CUDA_USE_NVIDIA_BINDING</code> once at the environment level</b></summary>

Setting the variable at the environment level saves you from repeating the two `os.environ` lines
at the top of every script and notebook. The effect is exactly the same; it is only a convenience.

```bash
# conda activation hook — applies on every `mamba activate mc_39`
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export NUMBA_CUDA_USE_NVIDIA_BINDING=1' \
    > "$CONDA_PREFIX/etc/conda/activate.d/numba_cuda_binding.sh"
mamba activate mc_39   # re-activate for it to take effect
```

For Jupyter / VSCode notebooks, also set it in the kernel, so it applies even when the kernel is
launched without shell activation. Add an `"env"` block to the kernel's `kernel.json` (locate it
with `jupyter kernelspec list`):

```json
{
  "argv": ["...python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "env": { "NUMBA_CUDA_USE_NVIDIA_BINDING": "1" }
}
```

</details>

# Documentation

Documentation is under construction.
