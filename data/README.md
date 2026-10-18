# Data Directory

This folder holds the UNSW-NB15 dataset files used for training and testing the intrusion detection model.

## Required Files

Place these two CSV files here before running the training pipeline:

```
data/
├── UNSW_NB15_training-set.csv   (~175,341 records)
└── UNSW_NB15_testing-set.csv    (~82,332 records)
```

## Download Instructions

1. Visit the official UNSW-NB15 dataset page:  
   **https://research.unsw.edu.au/projects/unsw-nb15-dataset**

2. Navigate to the **Downloads** section.

3. Download the following files (look under "CSV files" or similar):
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

4. Copy/move the downloaded files into this `data/` folder.

5. Verify file placement:

```powershell
# Run from the project root
dir data\
```

You should see both CSV files listed.

## Why These Files Are Not Included in the Repository

The dataset files can be hundreds of megabytes in size. Committing large binary/CSV files to a Git repository is not recommended practice. The files are excluded via `.gitignore`.

## Dataset Citation

> Moustafa, N., & Slay, J. (2015).  
> UNSW-NB15: A comprehensive data set for network intrusion detection systems.  
> In *2015 Military Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE.  
> https://doi.org/10.1109/MilCIS.2015.7348942

## Expected Schema

The CSV files are expected to contain columns including (but not limited to):

| Column | Type | Description |
|---|---|---|
| `id` | int | Record identifier (excluded from model) |
| `dur` | float | Connection duration |
| `proto` | str | Protocol type (e.g., tcp, udp) |
| `service` | str | Network service (e.g., http, ftp) |
| `state` | str | Connection state |
| `spkts` | int | Source-to-destination packet count |
| `dpkts` | int | Destination-to-source packet count |
| `sbytes` | int | Source-to-destination bytes |
| `dbytes` | int | Destination-to-source bytes |
| `sttl` | int | Source-to-destination TTL |
| `dttl` | int | Destination-to-source TTL |
| `attack_cat` | str | Attack category name (excluded — reveals attack type) |
| `label` | int | **Target**: 0 = Normal, 1 = Attack |

The actual columns in your files are validated automatically by the data loader.
