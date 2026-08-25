# Gefpan

A local restoration studio for worn photographs.

Gefpan repairs fade, dust, scratches, color cast, blur, and the quiet losses of a print that has been handled too long. Chemistry runs on your machine. The file never leaves it.

## Treatments

| Mode | What it does |
| --- | --- |
| **Auto** | Scratch repair, white balance, denoise, levels, local contrast, sharpen |
| **Vintage** | Old-print bath: neutralize yellow, lift bleach, inpaint damage |
| **Denoise** | Chroma smooth + edge-preserving luma (NLM when strength is high) |
| **Color** | Cast removal, auto levels, CLAHE, vibrance |
| **Sharpen** | Wiener-like high-frequency lift + unsharp mask |
| **Resolve** | 2× super-resolution (FSRCNN, with bicubic fallback) |
| **Portrait** | Face-aware: gentler on skin, stronger on the room |
| **Document** | Flatten uneven light, clean paper, raise ink |

## Run the studio

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gefpan samples          # build the worn demo prints
python -m gefpan serve            # http://0.0.0.0:8000
```

Or `python app.py`.

On first restore with **Resolve** / 2×, Gefpan downloads the small FSRCNN ×2 weights into `models/`.

## Command line

```bash
python -m gefpan restore worn.jpg -o print.jpg -m vintage -s 0.75
python -m gefpan restore scan.jpg -o scan-clean.jpg -m document --upscale
```

## How it works

No GPU. The pipeline is classical computer vision with one optional tiny CNN:

- damage mask from morphological black-hat / top-hat + Telea inpainting
- gray-world and LAB cast neutralization
- chroma-only Gaussian + bilateral / NLM on luma
- percentile levels, CLAHE, S-curve, vibrance
- OpenCV `dnn_superres` FSRCNN for 2×

Open a photograph in the studio, optionally add a **trace plate** (a cleaner print of the same sitting), restore. The trace lends color and fills tears; the worn plate keeps its drawing so the result stays a photograph, not a new picture.

Hold **Space** to peek at the worn plate. **D** downloads the print.

```bash
python -m gefpan restore worn.jpg -o print.jpg --trace clean.jpg
```
