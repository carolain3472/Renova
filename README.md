# Renova

Field-deployable classifier for four floating aquatic plant species, trained in
PyTorch and shipped to a Raspberry Pi Zero 2W as a quantised TFLite model.

Water hyacinth (*Eichhornia crassipes*) is among the world's most damaging
invasive aquatic plants — it blankets waterways, starves them of oxygen and
strangles fishing and navigation. Monitoring which species is spreading in a
given lagoon usually means sending someone to look. This project asks whether a
$15 computer with a camera can do that job on the shoreline, with no network.

## The four classes

| Species | Common name |
|---|---|
| *Eichhornia crassipes* | Common water hyacinth |
| *Lemna minor* | Common duckweed |
| *Monochoria korsakowii* | Heartleaf false pickerelweed |
| *Pistia stratiotes* | Water lettuce |

Dataset: [Mendeley Data `vz6z64nwby`](https://data.mendeley.com/datasets/vz6z64nwby/1)
— 1,790 original images (390 / 470 / 450 / 480 per class). Only the original
images are used; the pre-augmented copies shipped with the dataset are discarded
so that augmentation stays under the pipeline's control and never leaks between
splits.

## Architecture comparison

Six backbones were fine-tuned identically and compared on the same stratified
70/15/15 split (1,253 / 268 / 269 images), each in two phases: classifier head
first at lr 1e-3, then the full backbone at lr 1e-5.

| Model | Params | Size | Accuracy | F1 | Cohen's κ | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **ConvNeXt-Pico** | 8.5M | 32.6 MB | **99.63** | **99.65** | 0.9950 | 99.99 |
| ResNet-18 | 11.2M | 42.6 MB | 96.28 | 96.25 | 0.9504 | 99.83 |
| **MobileNetV3-Small** | **1.5M** | **5.8 MB** | 94.80 | 94.73 | 0.9305 | 99.64 |
| MobileNetV3-Large | 4.2M | 16.0 MB | 93.31 | 93.44 | 0.9105 | 99.23 |
| EfficientNet-B0 | 4.0M | 15.3 MB | 92.57 | 92.68 | 0.9007 | 99.33 |

The two finalists were then validated with **5-fold cross-validation** and a
held-out 20% test set they had never seen:

| Model | CV accuracy | CV F1 | Hold-out accuracy | Hold-out κ |
|---|---|---|---:|---:|
| ConvNeXt-Pico | 99.65 ± 0.54 | 99.67 ± 0.52 | 99.44 | 0.9925 |
| MobileNetV3-Small | 94.62 ± 0.65 | 94.68 ± 0.64 | 97.77 | 0.9702 |

### Why MobileNetV3-Small ships, despite losing

ConvNeXt-Pico is the better classifier by five points and the tables say so
plainly. It is also **5.6× larger** and too slow to run comfortably on a Pi Zero
2W's 512 MB of RAM. MobileNetV3-Small holds 97.77% on unseen data at 5.8 MB
before quantisation, which is the trade the deployment target actually demands.

Reporting both is the point. A benchmark that quietly drops the model that won
is not a benchmark.

## Pipeline

1. **Stratified split** of original images only, seed 42, no leakage between folds
2. **On-the-fly augmentation** with `albumentations`
3. **Two-phase transfer learning** — frozen backbone, then full fine-tune
4. **Evaluation** — accuracy, macro F1, precision, recall, Cohen's κ, ROC-AUC,
   confusion matrix, and **Grad-CAM** overlays to check the model looks at the
   plant rather than at the water
5. **Fine-tuning hook** for field photographs taken at the target lagoon, since
   a model trained on curated dataset images meets rather different lighting
6. **Export** PyTorch → ONNX → TFLite with INT8 quantisation and a
   representative dataset
7. **On-device inference** via [`deploy/infer_pi.py`](deploy/infer_pi.py),
   reading frames from `picamera2`

## Running it

```bash
git clone https://github.com/carolain3472/Renova.git
cd Renova
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Open [`notebooks/water_hyacinth_classifier.ipynb`](notebooks/water_hyacinth_classifier.ipynb).
Outputs are kept deliberately: the plots, confusion matrices and Grad-CAM
overlays are the evidence behind the tables above, and they render directly on
GitHub.

### On the Raspberry Pi

Hardware: Pi Zero 2W (or 3/4/5), Pi Camera v2 or v3, microSD 16 GB+.

```bash
sudo apt install -y python3-picamera2
pip install tflite-runtime numpy pillow

# Live camera loop — model and labels default to the filenames below
python deploy/infer_pi.py

# Or classify a single photograph
python deploy/infer_pi.py --image sample.jpg
```

Copy `water_hyacinth_int8.tflite` and `labels.json` (both written by the export
step of the notebook) next to the script, or point at them with `--model` and
`--labels`.

## Repository layout

```
notebooks/
  water_hyacinth_classifier.ipynb   training, benchmark, CV, export
deploy/
  infer_pi.py                       on-device inference loop
requirements.txt
```

Trained checkpoints are not committed — they are reproducible from the notebook,
and the `.pth` files run from 5.9 MB to 42.7 MB each.

## Known limitations

- Accuracy on curated dataset images is not accuracy in the field. Images taken
  at a real lagoon carry glare, motion blur and mixed species in one frame, none
  of which the benchmark captures — the fine-tuning step exists precisely because
  that gap is expected to be large
- Four species is a narrow world; a real deployment meets plants outside the
  label set and currently has no way to say "none of these"
- INT8 quantisation is measured for size, not yet for accuracy loss on device
- No automated tests and no CI

## License

MIT. The dataset keeps the terms of its Mendeley Data release.
