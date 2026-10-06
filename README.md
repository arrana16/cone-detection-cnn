# Cone colour classifier

This project prepares FSOCO cone boxes for offline colour classification. The current milestone defines the PyTorch input/output contract and formats a reproducible dataset manifest. It does not train a model or claim accuracy.

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
- `build_model(pretrained=True)` uses Torchvision's EfficientNet B0 ImageNet weights, downloading them if they are not cached. Pass `pretrained=False` for a random-weight shape check; its predictions are not meaningful until trained.

## Train the classifier head

`train.py` performs transfer learning. It loads the pretrained ImageNet EfficientNet B0 backbone, replaces its final layer with a fresh three-class layer, freezes the backbone, and trains only that final layer. It reads labelled rows from the `train` and `val` manifest splits, applies class-weighted cross-entropy, and never uses the `test` split. The best checkpoint is selected by validation macro F1.

Run a small balanced pilot similar to the interrupted experiment:

```bash
python3 train.py \
  --max-train-per-class 600 \
  --max-val-per-class 150 \
  --epochs 8 \
  --device auto \
  --output-dir runs/efficientnet_b0_pilot
```

For training on every labelled training row, omit both `--max-*-per-class` options. The default output directory is `runs/efficientnet_b0/`. During each training and validation pass, the command shows batch progress, running loss and accuracy, and estimated time remaining. Training writes `best_model.pt`, `history.csv`, and `config.json`; `.gitignore` excludes run artifacts and model weights. The first run downloads the pretrained weights if Torchvision has not cached them. Use `python3 train.py --help` to see all options. Training the output layer is an initial transfer-learning baseline; it does not guarantee that the model will generalize well to the full dataset or to camera footage.

## Validation

Manifest generation validates annotation parsing, image pairs and dimensions, rectangle geometry, and image bounds before writing output. The initial offline checks also verify the crop tensor shape and the model's three-logit output. These checks do not measure classification accuracy.
