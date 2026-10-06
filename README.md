# Cone colour classifier

This project prepares FSOCO cone boxes and trains offline cone-colour classifiers. It includes the original 96×96 EfficientNet workflow and a separate 32×32 comparison between EfficientNet V2 S transfer learning and an AMZ-inspired CNN.

## Set up the dataset

The FSOCO bounding-box training archive is about 24 GB and is not stored in Git. Download it from the [official FSOCO download page](https://fsoco.github.io/fsoco-dataset/download), then extract it so the project has this layout:

```text
data/
└── fsoco_bounding_boxes_train/
    ├── meta.json
    ├── ampera/
    │   ├── ann/
    │   └── img/
    ├── amz/
    │   ├── ann/
    │   └── img/
    └── ...
```

For example, if the downloaded archive is named `fsoco_bounding_boxes_train.zip` in `data/`, run:

```bash
mkdir -p data/fsoco_bounding_boxes_train
unzip data/fsoco_bounding_boxes_train.zip -d data/fsoco_bounding_boxes_train
```

The archive has the contributor folders and `meta.json` at its root, so extracting it into `data/fsoco_bounding_boxes_train/` creates the expected structure. If the archive was downloaded elsewhere, replace its path in the `unzip` command. The `/data/` rule in `.gitignore` excludes the archive, extracted dataset, and generated manifest files; each fresh checkout needs its own local dataset setup.

## Dataset manifest

The source data is expected under `data/fsoco_bounding_boxes_train/<contributor>/`, with `ann/` and `img/` directories. After setting it up as described above, run:

```bash
python3 prepare_manifest.py
```

This writes:

- `data/manifest.csv`: one row per annotated rectangle. Image and annotation paths are relative to the FSOCO data root. CSV tag columns contain JSON arrays.
- `data/manifest_summary.json`: source label counts, row counts, contributor assignments, and achieved image split proportions.

The default split keeps each contributor folder together, uses seed 42, and targets 80% training, 10% validation, and 10% test images. Since contributors have different numbers of images, actual ratios are approximate. Change the seed with `--seed`; change paths with `--data-root` and `--output`.

`blue_cone` maps to `blue`, `yellow_cone` to `yellow`, both orange classes to `other`, and `unknown_cone` has an empty `target_label`. Issue-tagged and tiny boxes remain in the manifest and are marked by `issue_flag` and `tiny_flag`. A box is marked tiny if either side is shorter than 8 pixels.

## PyTorch contract

`cone_classifier.py` defines `build_model`, `crop_from_box`, `preprocess_crop`, `predict_crop`, and `predict_box`.

- `ConeColorClassifier.forward` accepts normalized float tensors shaped `[batch, 3, 96, 96]` and returns raw logits ordered `blue`, `yellow`, `other`.
- `crop_from_box` extracts one RGB cone box with a 15% margin on each side of its longer dimension, pads at image borders with a neutral ImageNet-mean colour, and returns a square crop.
- `preprocess_crop` pads to square if needed, resizes bilinearly to 96×96, and applies ImageNet normalization.
- `predict_crop` returns the highest scoring label and all three softmax probabilities. `predict_box` combines box extraction and prediction. Neither applies an abstention threshold.
- `build_model(pretrained=True)` uses Torchvision's EfficientNet V2 S ImageNet weights, downloading them if they are not cached. Pass `pretrained=False` for a random-weight shape check; its predictions are not meaningful until trained.

Torchvision's pretrained EfficientNet V2 S recipe uses 384×384 images. This project currently keeps its 96×96 crop contract, so validate the trained model on the target data before relying on its accuracy.

## Train the classifier head

`train.py` performs transfer learning. It loads the pretrained ImageNet EfficientNet V2 S backbone, replaces its final layer with a fresh three-class layer, freezes the backbone, and trains only that final layer. It reads labelled rows from the `train` and `val` manifest splits, applies class-weighted cross-entropy, and never uses the `test` split. The best checkpoint is selected by validation macro F1.

Run a small balanced pilot similar to the interrupted experiment:

```bash
python3 train.py \
  --max-train-per-class 600 \
  --max-val-per-class 150 \
  --epochs 8 \
  --device auto \
  --output-dir runs/efficientnet_v2_s_pilot
```

For training on every labelled training row, omit both `--max-*-per-class` options. The default output directory is `runs/efficientnet_v2_s/`. During each training and validation pass, the command shows batch progress, running loss and accuracy, and estimated time remaining. Training writes `best_model.pt`, `history.csv`, and `config.json`; `.gitignore` excludes run artifacts and model weights. The first run downloads the pretrained weights if Torchvision has not cached them. Use `python3 train.py --help` to see all options. Training the output layer is an initial transfer-learning baseline; it does not guarantee that the model will generalize well to the full dataset or to camera footage.

## Validation

Manifest generation validates annotation parsing, image pairs and dimensions, rectangle geometry, and image bounds before writing output. The 32×32 experiment checks also verify the crop tensor shape, each model's three-logit output, and the AMZ intermediate dimensions. Shape and CLI checks do not measure classification accuracy; use the held-out test evaluation after training for that.

## 32×32 EfficientNet and AMZ experiments

The new experiments use source annotation coordinates and the existing square crop rule: add 15% context on each side of the longer box dimension, pad outside the source image with the ImageNet mean, resize bilinearly to 32×32, and apply ImageNet normalization. This is an FSOCO adaptation of AMZ's crop pipeline; the CNN layer sizes follow Figure 5 in `amz-planning (1).pdf`.

Create the separate blue/yellow/unknown manifest while retaining the legacy `data/manifest.csv` mapping:

```bash
python3 prepare_manifest.py \
  --label-scheme amz32 \
  --output data/manifests/amz32.csv
```

The `amz32` mapping sends `blue_cone` to `blue`, `yellow_cone` to `yellow`, and both orange classes plus `unknown_cone` to `unknown`. The same seed 42 contributor-separated splits are used. The new manifest summary is `data/manifests/amz32_summary.json`.

Both model entry points support `train` and `eval`. Training uses the train and validation splits, class-weighted cross-entropy, early stopping on validation macro F1, and an explicit output directory that must be empty. Use a small balanced pilot first:

```bash
python3 -m experiments.efficientnet_v2_s train \
  --output-dir runs/amz32/efficientnet_v2_s/pilot \
  --max-train-per-class 600 --max-val-per-class 150 --device auto

python3 -m experiments.amz_cnn train \
  --output-dir runs/amz32/amz_cnn/pilot \
  --max-train-per-class 600 --max-val-per-class 150 --device auto
```

For full training, omit both per-class limits and use separate run directories:

```bash
python3 -m experiments.efficientnet_v2_s train \
  --output-dir runs/amz32/efficientnet_v2_s/full --device auto

python3 -m experiments.amz_cnn train \
  --output-dir runs/amz32/amz_cnn/full --device auto
```

EfficientNet V2 S uses ImageNet weights and trains only its final classifier layer. The AMZ CNN trains from scratch. It uses same-padded convolutions with batch normalization, ReLU, 0.2 dropout, and max pooling; its `4×4×128` feature map flattens to 2048 values before the fully connected layers. Training defaults to 8 epochs, batch size 32, patience 3, and AdamW weight decay `1e-4`; learning rates are `0.003` for EfficientNet and `0.001` for AMZ CNN. In an interactive terminal, train and validation progress updates on one in-place line per phase; redirected logs receive periodic progress lines. `history.csv` and `config.json` are updated in the run directory, and the best checkpoint is `best_model.pt`.

To train the AMZ CNN for 16 more epochs from an existing best checkpoint, use a new output directory:

```bash
python3 -m experiments.amz_cnn train \
  --checkpoint runs/amz32/amz_cnn/full/best_model.pt \
  --output-dir runs/amz32/amz_cnn/continued \
  --epochs 16 --patience 16 --device auto
```

With `--checkpoint`, `--epochs` counts additional epochs after the checkpoint's saved epoch. The new run keeps its own `history.csv`, with epoch numbers continuing from that saved epoch, and retains the input checkpoint as `best_model.pt` until validation macro F1 improves. The checkpoint contains model weights but no optimizer state, so AdamW starts fresh. Use the same manifest, data root, seed, and per-class limits as the original run. `--patience 16` allows all 16 additional epochs even if validation does not improve.

The current `runs/amz32/amz_cnn/full/best_model.pt` was saved at epoch 6, although that run completed 8 epochs. To reach epoch 24 from this checkpoint, use `--epochs 18 --patience 18` instead.

Evaluate a chosen checkpoint on the held-out test split after training:

```bash
python3 -m experiments.efficientnet_v2_s eval \
  --checkpoint runs/amz32/efficientnet_v2_s/pilot/best_model.pt --split test

python3 -m experiments.amz_cnn eval \
  --checkpoint runs/amz32/amz_cnn/pilot/best_model.pt --split test
```

Evaluation writes `metrics_test.json` with accuracy, macro F1, per-class precision/recall/F1/support, and a confusion matrix, plus `predictions_test.csv` with each prediction and class probabilities. The test split is not used for training or checkpoint selection.

To follow epoch metrics or inspect a completed evaluation:

```bash
tail -f runs/amz32/efficientnet_v2_s/pilot/history.csv
cat runs/amz32/efficientnet_v2_s/pilot/metrics_test.json
```
