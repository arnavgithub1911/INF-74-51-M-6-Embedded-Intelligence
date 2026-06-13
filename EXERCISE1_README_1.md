# Exercise 1 — Data Exploration & Sliding Window Preprocessing

> *Before you can teach a machine to recognise running from walking, you need to understand what 100 sensor readings per second actually look like — and how to turn a river of numbers into something a model can learn from.*

---

## 🎯 Objective

This exercise introduces the **PAMAP2 Physical Activity Monitoring dataset** and establishes the preprocessing pipeline that all subsequent exercises build on. The core task is understanding the raw IMU sensor data, handling its imperfections, and transforming a continuous time-series stream into structured, labelled windows ready for machine learning.

---

## 🗃️ Dataset — PAMAP2

The PAMAP2 dataset captures physical activity data from **9 subjects** wearing three IMU sensors simultaneously — on the **hand**, **chest**, and **ankle**. Each sensor records at **100Hz**, meaning 100 data points every second.

### Sensor placement

```
        [HAND sensor]
             |
        [CHEST sensor]
             |
        [ANKLE sensor]
```

### Raw data structure

Each `.dat` file (one per subject) contains **54 columns + 1 timestamp**:

| Column(s) | Content |
|-----------|---------|
| 1 | Timestamp (seconds) |
| 2 | Activity ID (label) |
| 3 | Heart rate (bpm) |
| 4 – 20 | IMU Hand (17 channels) |
| 21 – 37 | IMU Chest (17 channels) |
| 38 – 54 | IMU Ankle (17 channels) |

### Each IMU block contains (17 channels):

| Channel(s) | Measurement | Detail |
|------------|-------------|--------|
| 1 | Temperature | °C |
| 2 – 4 | Accelerometer (high-g) | ±16g, 13-bit resolution |
| 5 – 7 | Accelerometer (low-g) | ±6g, 13-bit resolution |
| 8 – 10 | Gyroscope | rad/s |
| 11 – 13 | Magnetometer | μT |
| 14 – 17 | Orientation | ⚠️ Invalid in this collection — not used |

### Activity labels

| ID | Activity | ID | Activity |
|----|----------|----|----------|
| 0 | Other (transient) | 12 | Ascending stairs |
| 1 | Lying | 13 | Descending stairs |
| 2 | Sitting | 16 | Vacuum cleaning |
| 3 | Standing | 17 | Ironing |
| 4 | Walking | 18 | Folding laundry |
| 5 | Running | 19 | House cleaning |
| 6 | Cycling | 20 | Playing soccer |
| 7 | Nordic walking | 24 | Rope jumping |
| 9 | Watching TV | | |
| 10 | Computer work | | |
| 11 | Car driving | | |

---

## 🔧 Preprocessing Pipeline

The raw data cannot be fed directly into a model. Three preprocessing steps transform it into usable input:

### Step 1 — Column selection

From 54 raw channels, only **4 key accelerometer channels** are selected for this exercise:

- `col 4` — Hand accelerometer X
- `col 5` — Hand accelerometer Y  
- `col 6` — Hand accelerometer Z
- `col 7` — Hand accelerometer (low-g) X

These are the most informative channels for distinguishing physical activities and form the feature matrix **X**, while column 2 (Activity ID) forms the label vector **y**.

```python
cols = [4, 5, 6, 7]   # Selected accelerometer channels
X = dataset.iloc[:, cols].values
y = dataset.iloc[:, 1].values
```

### Step 2 — NaN handling

Sensor dropouts and transmission errors produce missing values scattered throughout the data. These cannot be left as-is — a NaN propagates through any computation it touches.

**Strategy: forward-fill imputation using `SimpleImputer`**

Each missing value is replaced with the most recent valid reading from the same channel. This preserves the temporal character of the signal better than mean imputation, which would introduce artificial flat regions.

```
NaN counts found across all subjects:

Subject 101:   4,362 NaN values → 0 after imputation
Subject 102:   8,187 NaN values → 0 after imputation
Subject 103:   1,566 NaN values → 0 after imputation
Subject 104:   6,642 NaN values → 0 after imputation
Subject 105:   4,623 NaN values → 0 after imputation
Subject 106:   3,063 NaN values → 0 after imputation
Subject 107:   4,518 NaN values → 0 after imputation
Subject 108:   6,453 NaN values → 0 after imputation
Subject 109:       9 NaN values → 0 after imputation
```

### Step 3 — Sliding window segmentation

A continuous time series cannot be fed directly to a classifier. Instead, it is segmented into fixed-length **overlapping windows**. Each window becomes one training sample.

```
Default configuration:
  Window size : 2.0 seconds  →  200 samples at 100Hz
  Step size   : 0.5 seconds  →  50 samples overlap stride
  Overlap     : 75%
```

**How windowing works:**

```
Raw signal (time →):
|----window 1----|
          |----window 2----|
                    |----window 3----|

← 200 samples →
     ← 50 step →
```

**Label assignment per window:** The majority activity label within the window is assigned as the window's label. This handles the transitions between activities gracefully.

**Resulting shapes after windowing:**

```
Subject 101:  376,417 raw rows  →  (7,439,  200, 4)  windows
Subject 102:  447,000 raw rows  →  (8,840,  200, 4)  windows
Subject 103:  252,833 raw rows  →  (4,990,  200, 4)  windows
Subject 104:  329,576 raw rows  →  (6,513,  200, 4)  windows
Subject 105:  374,783 raw rows  →  (7,406,  200, 4)  windows
Subject 106:  361,817 raw rows  →  (7,148,  200, 4)  windows
Subject 107:  313,599 raw rows  →  (6,191,  200, 4)  windows
Subject 108:  408,031 raw rows  →  (8,067,  200, 4)  windows
Subject 109:    8,477 raw rows  →     (159,  200, 4)  windows
```

Each processed subject is saved as a compressed `.npz` file:
- `arr_0` → X windows, shape `(n_windows, 200, 4)`
- `arr_1` → y labels, shape `(n_windows,)`

---

## 📊 Visualisations

> **To add your own plots here:**
> Export graphs from your notebook as `.png` files and place them in an `images/` folder inside this directory. Then uncomment and update the lines below.

```
<!-- ![Activity Distribution](images/activity_distribution.png) -->
<!-- ![Raw Accelerometer Signal](images/raw_signal.png) -->
<!-- ![Windowed Signal Example](images/windowed_signal.png) -->
<!-- ![NaN Distribution](images/nan_distribution.png) -->
```

**How to export a plot from your notebook:**
```python
plt.savefig("images/activity_distribution.png", dpi=150, bbox_inches='tight')
```

---

## 💡 Key Takeaways

- Raw sensor data is messy — nearly every subject had thousands of missing values that needed to be handled before any modelling could begin.
- Sliding windows with 75% overlap significantly increase the number of training samples, helping balance the dataset and give models more temporal context per sample.
- The choice of window size is a genuine tradeoff: longer windows capture more activity context but reduce temporal resolution and sample count.
- Only 4 of the 52 sensor channels were used here — channel selection is itself a design decision with significant downstream impact on model performance.

---

## ▶️ How to Run

1. Upload the PAMAP2 dataset to Kaggle and update `DATASET_PATH` in the notebook
2. Open `embedded-intelligence-ex-1.ipynb`
3. Run all cells in order

```bash
# Output files saved to:
Protocol/subject101.npz
Protocol/subject102.npz
...
Protocol/subject109.npz
```

---

*Exercise 1 of 6 · Embedded Intelligence · RPTU Kaiserslautern*  
*Dataset: PAMAP2 — Bo Zhou, DFKI · [UCI Repository](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)*
