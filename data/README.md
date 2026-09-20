# Data Directory

This directory stores the raw dataset used by EcoPulse AI.

## Dataset

**Name:** Appliances Energy Prediction  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction  
**File:** `energydata_complete.csv`

## Automatic Download

The dataset is downloaded automatically when you run the training script:

```bash
python src/train.py
```

The `data_loader.py` module will:
1. Check if `energydata_complete.csv` already exists locally.
2. If not, attempt to download it from the UCI repository (zip).
3. Fall back to a GitHub mirror if the primary source fails.

## Manual Download

If automatic download fails:

1. Visit: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
2. Download the zip file.
3. Extract `energydata_complete.csv` into this `data/` directory.

## Dataset Description

| Property      | Value                              |
|---------------|------------------------------------|
| Observations  | 19,735                             |
| Period        | Jan 11, 2016 – May 27, 2016       |
| Frequency     | Every 10 minutes                   |
| Location      | Low-energy house, Stambruges, Belgium |
| Target column | `Appliances` (Wh)                 |
| Features      | Indoor/outdoor temperature, humidity, lights, weather variables |

## Citation

> Candanedo, L., Feldheim, V., & Deramaix, D. (2017).  
> *Appliances energy prediction*.  
> UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G

**Note:** The dataset file is excluded from version control via `.gitignore` to keep the repository size small.
