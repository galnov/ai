# Guess What I'm Drawing: A Neural Network That Recognizes Drawings

Final project in software engineering with a specialization in artificial intelligence.

You draw with the mouse (or a finger on a touch screen), and a deep neural network guesses **while you draw** what
the drawing is, out of 42 categories: cat, bicycle, pizza, rainbow and more. The network was trained on hundreds of
thousands of real drawings by people from all over the world, from Google's Quick, Draw! dataset.

In the app:
* **Free drawing**: the guesses update with every stroke, showing the top 5 guesses and their confidence
* **"What the network sees"**: your drawing as it enters the network, 28 by 28 pixels
* **Game**: 6 drawings, 20 seconds each. At the end there's a summary with your drawings
* **What I know**: all the categories, with examples of how other people drew each one
* **Guess out loud**, if the browser supports speech
* **Help the network learn**: save your drawings as examples, and they are added to the next training run

---

## 1. Problem definition

**The problem:** given a quick, imprecise hand drawing, classify it as one of 42 categories.

**Type of problem:** multi-class image classification with supervised learning. Every drawing in the dataset comes
with the label of what the user was asked to draw.

**Why it's hard:**
* Everyone draws differently: a cat can be a face with ears, or a whole body with a tail
* The drawings are quick (up to 20 seconds in the original game), partial and sometimes unfinished
* Some categories are similar: cup and cake, sun and flower, sword and key
* The network has to guess even when the drawing isn't finished yet

**Why 28 by 28?** That's the dataset's resolution. It's enough to recognize the overall shape, and small enough that
training is fast and the network answers within milliseconds.

---

## 2. Project structure

```
doodle-ai/
├── requirements.txt
├── data/
│   ├── raw/              the downloaded drawings (created by the download)
│   └── custom/           drawings saved from the app
├── src/
│   ├── config.py         the categories (dataset name, display name, emoji), paths and amounts
│   ├── download_data.py  collecting the data from Quick, Draw!
│   ├── data.py           splitting into training, validation and test
│   ├── explore_data.py   data exploration and plots
│   ├── render.py         turning a drawing (strokes) into a 28×28 image, exactly like the dataset
│   ├── model.py          the network and data augmentation
│   ├── train.py          training
│   ├── evaluate.py       evaluation on the test set
│   ├── predictor.py      using the network inside the app
│   ├── export_web.py     exporting the trained network for the web page (GitHub Pages)
│   └── import_drawings.py  adding drawings saved on the web page to the training data
├── app/
│   ├── server.py         Flask server
│   ├── templates/        the user interface (HTML)
│   └── static/           styling (CSS) and client-side code (JavaScript)
├── docs/                 the same app as a static web page, for GitHub Pages (section 7)
│   ├── cnn.js            the CNN in plain JavaScript
│   ├── render.js         drawing -> 28×28 image, the twin of src/render.py
│   ├── worker.js         runs the network in a background thread
│   ├── backend.js        does in the browser what the Flask server does
│   └── model/            the exported network (created by  python -m src.export_web)
├── notebooks/            training in Google Colab
├── models/               the trained network (created by training)
└── reports/              plots and reports (created automatically)
```

---

## 3. Installation and running

You need Python 3.10 to 3.13.

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt
```

### Step A: Collect data

```bash
python -m src.download_data
```

The command downloads 10,000 drawings for each of the 42 categories, about 330MB in total. Each category has a
huge file in the dataset with more than 100,000 drawings, but the script reads only the beginning of each file and
closes the connection. If the download is interrupted, run it again; what was already downloaded isn't downloaded again.

> Want a faster download and training? `python -m src.download_data --per-class 4000`.
> Accuracy will be slightly lower.

### Step B: Explore the data

```bash
python -m src.explore_data
```
Creates three plots in `reports/`: examples from each category, the "average drawing" of each category, and how much
"ink" the drawings of each category use (from simple to complex).

### Step C: Training

```bash
python -m src.train --quick      # check that everything works (saved to a separate file, not for the app!)
python -m src.train              # full training: 15 epochs
```

On a Mac with Apple silicon, or with an NVIDIA graphics card, training uses the graphics card's acceleration automatically.

> **Training in Google Colab:** if training on your computer is slow, you can train for free on a T4 GPU with the
> notebook `notebooks/train_colab.ipynb`. Upload the notebook to Colab, choose T4, upload `doodle-ai.zip`
> and run the cells in order. At the end, download `doodle_cnn.pt` into the `models/` folder on your computer.

After every epoch the validation accuracy is printed: top-1 (the first guess is correct) and top-3 (the correct answer
is among the first 3 guesses). The best network is saved to `models/doodle_cnn.pt`.

### Step D: Evaluation

```bash
python -m src.evaluate
```

| File | What you see |
|---|---|
| `test_report.txt` | top-1 and top-3 accuracy, accuracy per category, and the pairs that get confused most |
| `confusion_matrix.png` | a 42 by 42 confusion matrix |
| `per_class_accuracy.png` | which categories are easy and which are hard |
| `mistakes.png` | examples of mistakes. Many of them are understandable: a person would be confused too |
| `training_curves.png` | loss and accuracy during training |

### Step E: The app

```bash
python -m app.server
```
and open in the browser: **http://localhost:5000**

---

## 4. Collecting and processing the data

### The dataset
Quick, Draw! is a Google game where people were asked to draw something within 20 seconds. It produced an open
dataset of 50 million drawings in 345 categories. We chose 42 categories that are familiar, different from each
other and fun to draw.

### The critical part: processing the user's drawing exactly like the dataset
The network learns from images that went through a specific processing. If the drawing in the app goes through
different processing (a thicker line, a drawing that isn't centered, different margins), the network will "see"
something different from what it learned, and recognition will suffer. So `render.py` reproduces exactly the process
Google documented:

1. **Alignment and scale:** move the drawing to the corner and scale it so the largest dimension is 255
2. **Simplification:** each stroke is simplified with the Ramer-Douglas-Peucker algorithm (keep only the points that matter for the shape)
3. **Rendering:** center, add margins and draw a line of width 16 (relative to a size of 256) with round caps
4. **Shrinking to 28×28**, with pixel averaging, so the edges are smoothed

This way a drawing made in the corner of the page, small or large, looks to the network exactly like the dataset
drawings. In "What the network sees" in the app you can see the result.

### Splitting
From each category: 8,000 drawings for training, 1,000 for validation (choosing the best model) and 1,000 for testing
(the final evaluation, on drawings that weren't used for any decision).

---

## 5. The network

| Layer | Output | Description |
|---|---|---|
| Input | 1 × 28 × 28 | a grayscale drawing: white lines on a black background |
| Block 1 | 32 × 14 × 14 | two 3×3 convolutions with BatchNorm and ReLU, then MaxPool and Dropout. Learns lines and edges |
| Block 2 | 64 × 7 × 7 | corners, circles, arcs |
| Block 3 | 128 × 3 × 3 | "parts": ears, wheels, leaves, a handle |
| Fully connected layer | 256 | with BatchNorm, ReLU and Dropout 0.4 |
| Output | 42 | a score for each category. softmax turns the scores into probabilities |

About 600 thousand parameters. Enough for the problem, and small enough to answer instantly while drawing.

### Design decisions

| What | Why |
|---|---|
| **Data augmentation** | in every batch each drawing is randomly rotated, shifted, zoomed in or out and sometimes mirrored. A crooked cat is still a cat. Computed on the whole batch at once with `affine_grid` |
| **BatchNorm** | stabilizes training and allows a high learning rate |
| **Dropout** | prevents "memorizing" the training set |
| **OneCycle LR** | the learning rate rises at first ("warm-up") and decreases gradually. Converges quickly and stably |
| **AdamW** | an adaptive optimizer with weight decay |
| **Label smoothing 0.1** | some drawings in the dataset are bad or unfinished. The network learns not to be "100% sure" |
| **Keeping the best model** | by validation accuracy |
| **Test-time augmentation** | at inference: average of the prediction for the drawing and its mirror image |

### Expected results
With 42 categories and 8,000 training drawings per category, a network like this usually reaches a top-1 accuracy
of about 85% to 90%, and top-3 above 95%. The easy categories are those with a unique shape (rainbow, scissors,
ladder), and the hard ones are similar to each other (cup and cake, cloud and sun). The confusion matrix in
`reports/` will show you the exact picture.

---

## 6. How the app works

```
Browser (JavaScript)                            Server (Python / Flask)
────────────────────                            ───────────────────────
drawing with the mouse: collect strokes [[x...],[y...]]
after every stroke (and every 0.35 s while drawing):
            ── POST /api/predict (the strokes) ──►  render.py: align, simplify, draw 28×28
                                                   the network: softmax of 42 probabilities
            ◄── top 5 guesses + the 28×28 image ──
speech bubble, confidence bars, "what the network sees"
game: is the first guess the word that was asked for?
```

We send the server the **strokes** and not an image, so the server can process them exactly like the dataset drawings.

---

## 7. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free, at an address like `https://<your-user>.github.io/doodle-ai/`.
It can't run Python, so the `docs/` folder holds a version of the app that needs no server: the drawing is
turned into a 28 × 28 image in JavaScript (`docs/render.js`, the twin of `src/render.py`), and the trained CNN
runs in the visitor's browser, in plain JavaScript (`docs/cnn.js`), in a background thread so drawing stays
smooth. The free drawing, the game, the voice and "What I know" all work the same.

**1. Export the trained network** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (the network, about 2.4 MB), `model.json` (the layers, categories
and accuracy) and `samples.json` (12 real drawings per category, if the dataset is downloaded).
BatchNorm is folded into the layers before it (in evaluation mode it is just a fixed scale and shift), so
the browser computes exactly the same network with fewer steps. Training on Colab? The last cell of the
notebook exports it too: extract `docs_model.zip` into the project folder.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000. (Opening `docs/index.html` directly as a file does not work: browsers
don't allow the background thread and the model files from `file://`.)

**3. Publish:**
1. Create a new repository on GitHub, for example `doodle-ai`
2. Push the project to it (`data/raw/` is in `.gitignore`, so the big dataset files are not uploaded;
   `docs/model/` must be included):
   ```bash
   git init
   git add .
   git commit -m "Guess What I'm Drawing"
   git branch -M main
   git remote add origin https://github.com/<your-user>/doodle-ai.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch →
   Branch: `main`, folder: `/docs` → Save**
4. After a minute or two the page is live at `https://<your-user>.github.io/doodle-ai/`

**What is different on the web page:**
* "Save as a training example" keeps the drawings in the visitor's browser. "Download my drawings" saves
  them to `my_drawings.json`, and `python -m src.import_drawings my_drawings.json` adds them to
  `data/custom/` for the next training run
* After training a better network, run `python -m src.export_web` again and push

**Checked during development:** the JavaScript CNN gives the same scores as the network computed with
PyTorch's rules, to within 0.000001 (float rounding only), and the JavaScript drawing-to-image conversion
matches the Python one to within a few percent of brightness at the edges of the lines.

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| A `CERTIFICATE_VERIFY_FAILED` error while downloading (Mac) | `pip install certifi` and run again, or run `Install Certificates.command` from the Python folder |
| "categories are missing" | the download didn't finish. Run `python -m src.download_data` again |
| The app says the model isn't ready | you need to train (`python -m src.train`) and restart the server |
| Recognition is weak | check that a full training ran, not `--quick`. Draw in the middle of the page, with clear lines, like the dataset drawings in the "What I know" tab |
| Speaking out loud doesn't work | depends on the browser and the operating system: an English voice must be installed. It usually works in Chrome and Safari |

---

## 9. Ideas for extensions

* **More categories:** add rows to `CATEGORIES` in `config.py`, download and train again
* **Fine-tuning on your own drawings:** collect 20 to 30 drawings per category and check whether accuracy on them improves
* **A network that uses the order of the strokes** (an RNN or a Transformer on the strokes themselves), and a comparison with the image network
* **Visual explanation:** Grad-CAM, to see which strokes the network "looked at"
* **Touch-screen and tablet support** already exists. You can show the app to the class on a big screen
