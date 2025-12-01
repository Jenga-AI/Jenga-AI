# MLflow UI Guide - Monitoring Training Progress

## Quick Start

**Access MLflow UI**: Open your browser and go to:
```
http://localhost:5000
```

## Step-by-Step Guide

### 1. View Your Experiment

When you first open MLflow UI:

1. You'll see the **Experiments** page
2. Look for experiment: **`QA_Test_Training`**
3. Click on it to see all runs

### 2. View a Training Run

In the experiment view:

1. You'll see a table with all runs (you should have 1 active run)
2. The active run will show:
   - **Status**: `RUNNING` (or `FINISHED` when done)
   - **Start Time**: When training started
   - **User**: Your username
3. **Click on the run name** to open detailed view

### 3. View Metrics Charts

Once inside the run:

#### **Automatic Metrics Tab**
1. Click on **"Metrics"** tab (should be already selected)
2. You'll see charts for all logged metrics:
   - `train_loss`
   - `val_loss`
   - `opening_accuracy`, `opening_precision`, `opening_recall`, `opening_f1_score`
   - `listening_accuracy`, `listening_precision`, `listening_recall`, `listening_f1_score`
   - (same for other heads: proactiveness, resolution, hold, closing)

#### **Chart Features**
- **X-axis**: Shows epoch/step number
- **Y-axis**: Metric value
- **Hover**: Shows exact values
- **Zoom**: Click and drag to zoom in
- **Download**: Click download icon to save chart

### 4. Compare Metrics

**To compare multiple metrics on one chart:**

1. Go back to the experiment view (click experiment name in breadcrumb)
2. **Select multiple runs** using checkboxes (if you run training multiple times)
3. Click **"Compare"** button at the top
4. You'll see:
   - Side-by-side comparison tables
   - Overlaid metric charts
   - Parameter differences

### 5. Custom Chart Setup

**To create custom visualizations:**

#### Option A: In Run View
1. Click **"Metrics"** tab
2. Click **"Add Chart"** button
3. Select metrics you want to visualize
4. Choose chart type (line, bar, scatter, etc.)
5. Customize axes and appearance

#### Option B: Compare Multiple Metrics
1. In the Metrics tab, find the metric selector
2. Use **Ctrl+Click** (or **Cmd+Click** on Mac) to select multiple metrics
3. They'll all plot on the same chart for easy comparison

### 6. Monitor Real-Time Progress

**While training is running:**

1. Keep the browser tab open on the run's **Metrics** page
2. **Refresh the page** periodically (press `F5` or click refresh)
   - MLflow updates metrics only when you refresh
   - Charts will automatically update with new data points
3. Set up **auto-refresh** (if your browser supports it):
   - Chrome: Use extension like "Auto Refresh Plus"
   - Firefox: Use "Tab Reloader" extension

### 7. Key Metrics to Watch

For your QA training, focus on these charts:

| Metric | What to Look For |
|--------|------------------|
| `train_loss` | Should decrease steadily |
| `val_loss` | Should decrease (watch for overfitting if it increases) |
| `*_accuracy` | Should increase over epochs |
| `*_f1_score` | Overall performance measure - higher is better |

**Example Chart Layout**:
```
┌─────────────────────┐  ┌─────────────────────┐
│   train_loss        │  │   val_loss          │
│   (decreasing ↓)    │  │   (decreasing ↓)    │
└─────────────────────┘  └─────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐
│  All heads F1       │  │  All heads accuracy │
│  (increasing ↑)     │  │  (increasing ↑)     │
└─────────────────────┘  └─────────────────────┘
```

### 8. View Parameters

To see training configuration:

1. In the run view, click **"Parameters"** tab
2. You'll see all the config values:
   - `base_model`: "distilbert-base-uncased"
   - `batch_size`: 2
   - `learning_rate`: 3.0e-5
   - `num_epochs`: 3
   - etc.

### 9. Download Charts

**To save a chart as an image:**

1. Hover over any chart
2. Click the **camera icon** (📷) in the top-right corner
3. Save as PNG

**To export metrics data:**

1. Click **"Download CSV"** button (if available)
2. Or use MLflow API:
   ```python
   import mlflow
   runs = mlflow.search_runs(experiment_names=["QA_Test_Training"])
   runs.to_csv("training_metrics.csv")
   ```

### 10. Common Issues

**⚠️ Charts not updating?**
- Refresh the page manually (F5)
- Check if training is still running
- Look at the terminal for errors

**⚠️ Can't access UI?**
- Make sure MLflow server is running
- Check if port 5000 is already in use
- Try `http://127.0.0.1:5000` instead

**⚠️ Run not showing up?**
- Wait a few seconds after training starts
- Refresh the experiments page
- Check terminal for MLflow errors

## Quick Reference Commands

```bash
# Start MLflow UI
cd /home/collins/Documents/Jenga-AI/Jenga-AI
.venv/bin/mlflow ui --backend-store-uri ./mlruns --port 5000

# Check if server is running
curl http://localhost:5000

# Stop MLflow UI (if needed)
# Press Ctrl+C in the terminal where it's running
```

## Screenshot Example (What You'll See)

```
┌─────────────────────────────────────────────────┐
│ MLflow                              [Search] 🔍 │
├─────────────────────────────────────────────────┤
│ Experiments > QA_Test_Training > Run abc123     │
├─────────────────────────────────────────────────┤
│ [Parameters] [Metrics] [Artifacts] [Tags]       │
├─────────────────────────────────────────────────┤
│                                                 │
│  📊 train_loss                                  │
│     ╱╲                                         │
│    ╱  ╲___                                     │
│   ╱       ╲___                                 │
│  ╱            ╲___                             │
│  0────1────2────3  (epochs)                    │
│                                                 │
│  📊 val_loss                                    │
│     ╱╲                                         │
│    ╱  ╲___                                     │
│   ╱       ╲___                                 │
│  0────1────2────3  (epochs)                    │
│                                                 │
│  [+ Add Chart]                                  │
└─────────────────────────────────────────────────┘
```

## Pro Tips

1. **Compare Loss vs Accuracy**: Plot both on same chart to see correlation
2. **Watch for Overfitting**: If `val_loss` increases while `train_loss` decreases, you're overfitting
3. **Per-Head Analysis**: Compare F1 scores across different heads to see which tasks perform best
4. **Bookmark the Run**: Save the run URL for easy access later
5. **Export Before Closing**: Download metrics CSV before stopping the server

---

**Need Help?**
- MLflow Docs: https://mlflow.org/docs/latest/tracking.html
- Current UI: http://localhost:5000
