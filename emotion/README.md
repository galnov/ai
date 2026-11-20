# Emotion Mirror: Recognizing Emotions from Facial Expressions with Deep Learning

Final project in software engineering with a specialization in artificial intelligence.

A convolutional neural network (CNN) trained on about 35,000 face images recognizes, in real time from the webcam,
which emotion a face expresses: angry, disgust, fear, happy, neutral, sad or surprise. The app runs in the
browser and shows:

* A frame around each face with the emotion and the confidence level
* Probability bars for each of the seven emotions
* **A heat map (Grad-CAM)** that shows which part of the face the network "looked at" when it decided
* A timeline of the emotions over the last 30 seconds
* **A "copy the emotion" challenge**: a 5-round game where each round asks you to show a specific emotion
* Uploading a photo from your computer and taking a screenshot of the result

---

## 1. Problem definition

**The problem:** given an image of a person's face, classify the facial expression as one of 7 basic emotions.

**Type of problem:** multi-class classification with supervised learning. Every training image comes with a correct label.

**Why it's hard:**
* Different people express the same emotion differently
* Some emotions look very similar (fear and surprise, anger and disgust)
* Shooting conditions vary: lighting, angle, glasses, hair
* The data is imbalanced: about 9,000 images of "happy" but only about 550 of "disgust"
* It's hard for humans too: human accuracy on FER-2013 is estimated at about 65%

**Uses:** user-experience research, games that react to the player, helping people who struggle to recognize emotions.

---

## 2. Project structure

```
emotion/
├── requirements.txt          required libraries
├── data/                     the data (see data/README.md)
│   ├── fer2013/train, test   the dataset from Kaggle
│   └── custom/               images you collected yourself
├── models/emotion_cnn.pt     the trained model (created by training)
├── reports/                  plots and reports (created automatically)
├── notebooks/train_colab.ipynb   training in Google Colab on a GPU
├── src/
│   ├── config.py             settings: emotions, colors, paths
│   ├── data_sources.py       finding and loading the data from disk
│   ├── dataset.py            PyTorch Dataset and data augmentation
│   ├── model.py              network definition
│   ├── train.py              training
│   ├── evaluate.py           evaluation on the test set
│   ├── gradcam.py            visual explanation (Grad-CAM)
│   ├── explore_data.py       data exploration and plots
│   ├── collect_data.py       collecting images from the webcam
│   ├── vision.py             face detection and image processing (OpenCV)
│   ├── predictor.py          using the network on a new image
│   └── export_web.py         packs the trained network for the web page (docs/model/)
├── app/
│   ├── server.py             Flask server
│   ├── templates/index.html  the user interface
│   └── static/               styling (CSS) and client-side code (JavaScript)
└── docs/                     the web version for GitHub Pages: the same app, with no server (section 7)
```

---

## 3. Installation

You need Python 3.10 to 3.12. Run from the project folder:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> If you have an NVIDIA graphics card, install PyTorch with CUDA support following the instructions at https://pytorch.org.
> On a Mac with Apple silicon, the code automatically uses MPS acceleration.

---

## 4. Running, step by step

### Step A: Collect data

1. Download FER-2013 and arrange it as explained in [data/README.md](data/README.md).
2. **Optional but recommended:** collect images of yourself and your friends:
   ```bash
   python -m src.collect_data
   ```
   Keys 1 to 7 save the face as the matching emotion. Space toggles continuous saving. q quits.

### Step B: Explore the data

```bash
python -m src.explore_data
```
Creates three plots in `reports/`: the number of images per emotion, examples of each emotion, and the
"average face" of each emotion.

### Step C: Train

```bash
python -m src.train --quick     # check that everything works (under a minute). Does NOT train the app's model!
python -m src.train             # full training
python -m src.train --width 32  # smaller, faster network, for a computer without a GPU
```

> **Important:** `--quick` only checks that the code runs. It uses 2,000 images and a tiny network, and it is
> saved to a separate file (`emotion_cnn_quick.pt`) that the app does not load. For the app to work well, you
> must run the full training.
>
> At the start of training a table with the number of images is printed. The `train` row should show about
> 25,000 images. If there are far fewer, the dataset is not in the right place.

After every epoch the loss and accuracy on the training and validation sets are printed. The best model is saved
automatically to `models/emotion_cnn.pt`, and the plot `reports/training_curves.png` is updated.

**How long does it take?** It depends a lot on the computer. With a GPU, about 20 to 40 seconds per epoch.
On a CPU only, a few minutes per epoch, so full training can take several hours.
A convenient option: train in Google Colab with a free GPU (`notebooks/train_colab.ipynb`),
download the file `emotion_cnn.pt` and put it in the `models/` folder.

### Step D: Evaluate

```bash
python -m src.evaluate
```
Creates in `reports/`:
* `test_report.txt`: accuracy, precision, recall and F1 per emotion
* `confusion_matrix.png`: confusion matrix, showing which emotion gets confused with which
* `predictions_grid.png`: examples from the test set (green = correct prediction, red = wrong)
* `gradcam_grid.png`: heat maps showing what the network looked at

### Step E: The app

```bash
python -m app.server
```
and open in the browser: **http://localhost:5000**

Click "Start camera" and allow camera access. You can also drag a photo into the window.

---

## 5. The network

### Architecture

| Layer | Output (channels × height × width) | Description |
|---|---|---|
| Input | 1 × 48 × 48 | grayscale face image |
| Block 1 | 64 × 24 × 24 | two 3×3 convolutions with BatchNorm and ReLU, then MaxPool and Dropout |
| Block 2 | 128 × 12 × 12 | same structure |
| Block 3 | 256 × 6 × 6 | same structure |
| Block 4 | 512 × 3 × 3 | same structure |
| Global Average Pooling | 512 | average of each feature map |
| Fully connected layer | 256 | with BatchNorm, ReLU and Dropout 0.4 |
| Output | 7 | a score per emotion. softmax turns the scores into probabilities |

The network has about 4.8 million parameters (with `--width 64`). The first layers learn to detect edges and lines,
and the deeper ones learn complex shapes such as a smiling mouth, raised eyebrows or wide-open eyes.

### Design decisions

| What | Why |
|---|---|
| **3×3 convolutions in pairs** | two 3×3 layers see a 5×5 area with fewer parameters, and add another non-linearity between them |
| **Batch Normalization** | stabilizes training and allows a higher learning rate |
| **Dropout** | randomly "switches off" neurons during training to prevent overfitting |
| **Global Average Pooling** | replaces a huge fully connected layer, so fewer parameters and less overfitting |
| **Data augmentation** | mirror flip, rotation, cropping, lighting changes and partial occlusion. The network learns to cope with variations |
| **Class weights** | "disgust" has 16 times fewer images than "happy", so a mistake on a rare emotion costs more |
| **Label smoothing 0.1** | stops the network from being "100% sure", and helps with data where some labels are wrong |
| **AdamW** | an adaptive optimizer with weight decay (regularization) |
| **ReduceLROnPlateau** | when accuracy stalls, the learning rate is halved |
| **Early stopping** | stop if there's no improvement for 8 epochs, and keep the best model |
| **Test-time augmentation** | at inference, average the prediction for the image and its mirror image |

### Grad-CAM

To understand **why** the network decided what it decided, we compute the gradient of the chosen emotion's score
with respect to the feature maps of the last convolution layer. Maps with a high gradient are "important". Their
weighted sum gives a heat map, which shows for example that for "happy" the network looks at the mouth, and for
"surprise" at the eyes and eyebrows.

### Expected results

On FER-2013, a network like this usually reaches **63% to 68%** accuracy on the test set. That's around human
accuracy on this dataset. "Happy" and "surprise" are recognized best (80% and above). "Fear" and "sad" get confused
more. The confusion matrix in `reports/` will show you the exact picture.

---

## 6. How the app works

```
Browser (JavaScript)                        Server (Python / Flask)
────────────────────                        ───────────────────────
camera ← getUserMedia
capture a 640px frame ← canvas
        ── POST /api/predict (JPEG) ──►     decode the image
                                            face detection (OpenCV Haar Cascade)
                                            crop and resize to 48×48 grayscale
                                            network ← softmax ← probabilities
                                            (Grad-CAM ← PNG heat map)
        ◄── JSON: face positions, probabilities ──
draw frames, labels, heat map
update bars, timeline, game
```

---

## 7. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free. It can't run Python, PyTorch or OpenCV, so the `docs/` folder holds
a version of the app that needs no server: everything the server did happens in the visitor's browser.
**The camera images never leave the visitor's computer**: there is no server to send them to.

```
Browser only (JavaScript)
─────────────────────────
camera ← getUserMedia → capture a 640px frame ← canvas
        ── to a background thread (emotion-worker.js) ──►  face detection (faces.js: OpenCV's LBP cascade)
                                                           crop and resize to 48×48 grayscale
                                                           the CNN (emotion-cnn.js) ← softmax ← probabilities
                                                           (Grad-CAM, computed by hand ← heat map)
        ◄── face positions, probabilities ──
draw frames, labels, heat map, bars, timeline, game
```

**1. Export the trained network** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (the network, about 4.8 MB at width 64) and `model.json` (the layers,
the emotions and the accuracy). BatchNorm is folded into the layer before it (in evaluation mode it is just a
fixed scale and shift), so the browser computes exactly the same network with fewer steps. Training on
Colab? The last cell of the notebook exports it too: extract `docs_model.zip` into the project folder.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000. (Opening `docs/index.html` directly as a file does not work: browsers don't
allow the background thread and the model files from `file://`. The camera works on `http://localhost` and on
`https://` addresses, such as GitHub Pages.)

**3. Publish:** this project lives in the shared repository of all the projects. Push the repository, and
its workflow publishes this folder's `docs/` at `https://<your-user>.github.io/<repository>/emotion/` (the
README at the top of the repository explains the one-time setup). Don't push `data/`: the dataset is large,
and the images you collected yourself are photos of people (the repository's `.gitignore` leaves it out).

**What is different on the web page:**
* **Face detection:** the Flask app uses OpenCV's Haar cascade. The web page uses OpenCV's **LBP cascade**
  (`docs/face_cascade.json`, converted from OpenCV's `lbpcascade_frontalface.xml`), written in JavaScript
  (`docs/faces.js`): the same Viola-Jones idea (a 24 × 24 window at many sizes, a cascade of stages that
  rejects most windows after a few checks), with "Local Binary Pattern" features instead of Haar features:
  each feature compares 8 blocks of pixels with the block in the middle, giving an 8-bit code
* **Speed:** the network runs in plain JavaScript, so it is slower than PyTorch. With the camera it uses one
  pass per face (a few frames per second on a laptop); for a photo it also adds the mirror image, like the
  server. A network trained with `--width 32` is about 4 times faster

**Checked during development:** the JavaScript network gives the same probabilities as the network computed
with PyTorch's rules to within 0.000001, with and without the mirror image, and the same Grad-CAM map to
within 0.00001. The JavaScript face detector found 11 of 12 faces from the Labeled Faces in the Wild
dataset, no faces in 12 images without faces, and the same face as scikit-image's implementation of the
cascade in a test photo.

---

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| "No training data found" | check that the folders `data/fer2013/train` and `data/fer2013/test` exist |
| "Face detector file not found" | `pip install "opencv-python>=4.8,<5"` |
| The app says "Model not loaded" | you need to train first (`python -m src.train`) and restart the server |
| The camera doesn't open | open via `http://localhost:5000` (not an IP address), and allow camera access |
| "No face detected" | look straight at the camera, in good light, without strong light behind you |
| Training hangs at the start on Windows | add `--workers 0` |
| Training is too slow | `--width 32`, or train in Colab |

---

## 9. Ideas for extensions

* Transfer learning: compare with a pretrained network (ResNet18 or MobileNet) and show the differences in the report
* Calibration: check whether "80% confidence" really means the network is right 80% of the time
* A more accurate face detector (a DNN instead of Haar)
* Fine-tuning on the images you collected, and comparing accuracy before and after
