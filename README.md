# AFL SuperCoach Player Data & Prediction System

An AFL SuperCoach fantasy football analytics system that uses machine learning to predict player performance and provide data-driven insights for weekly team selection decisions.

## Overview

This project scrapes historical AFL player statistics, trains machine learning models on the data, and generates weekly predictions for SuperCoach player scores. The system automates the entire workflow from data collection to prediction delivery via Google Sheets integration.

## What It Does

**Core Purpose**: Predicts AFL player SuperCoach scores for upcoming rounds using Random Forest models trained on historical performance data.

### Key Features

- **Automated Data Collection**: Scrapes player statistics from FanFooty.com.au and other sources
- **Feature Engineering**: Creates 22+ predictive features including rolling averages, opponent strength ratings, and player role encodings
- **Machine Learning Models**: Three ensemble models for different prediction types
  - Regression model for exact score predictions
  - Binary classifier for 80+ point predictions
  - Binary classifier for 100+ point predictions
- **Weekly Predictions**: Generates ranked player predictions with opponent matchup information
- **Google Sheets Integration**: Automatically uploads predictions for easy team management access
- **Draft Preparation Tools**: Heat maps and analysis notebooks for pre-season draft planning
- **League Tracking**: Monitors transactions, team rosters, and ladder positions

## Prerequisites

### Required Software
- Python 3.7+
- pip (Python package manager)

### Required Python Packages
```
pandas
numpy
scikit-learn
requests
beautifulsoup4
gspread
google-auth
matplotlib
seaborn
jupyter
joblib
```

### External Dependencies
- Google Cloud Service Account (for Google Sheets integration)
- Active internet connection for web scraping

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sc-player-data
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Sheets authentication**
   - Create a Google Cloud Project
   - Enable Google Sheets API
   - Create a service account and download credentials
   - Save credentials as `service_account.json` in the project root
   - Share your target Google Sheet with the service account email

## Usage

### Weekly Automated Pipeline

Run the complete weekly workflow after each round completes:

```bash
python run_pipeline.py --round <completed_round_number>
```

This command will:
1. Ingest raw data for the completed round
2. Process data into structured format
3. Update master historical file
4. Retrain all models with latest data
5. Generate predictions for next round
6. Upload predictions to Google Sheets

### Individual Components

**Data Ingestion**
```bash
python ingest_fanfooty.py --year <year> --round <round>
```

**Feature Engineering**
```bash
python build_features.py --year <year> --round <round>
```

**Model Training**
```bash
python train_model.py --test-year <year>
```

**Generate Predictions**
```bash
python run_predictions.py --year <year> --round <round>
```

**Upload to Google Sheets**
```bash
python upload_to_gsheet.py --predictions-file <path_to_predictions.csv>
```

### Jupyter Notebooks

Interactive analysis and exploration:

- **`scrape-fanfooty.ipynb`** - FanFooty data scraping
- **`scrape-sc.ipynb`** - SuperCoach.com.au scraping
- **`predict-fanfooty.ipynb`** - Generate predictions interactively
- **`in-season.ipynb`** - In-season analysis and tracking
- **`draft_prep/SC 2025/`** - Pre-draft player analysis and heat maps

## Project Structure

```
sc-player-data/
├── README.md
├── requirements.txt
├── service_account.json          # Google Sheets credentials (not in repo)
│
├── Pipeline Scripts
│   ├── run_pipeline.py           # Main automated workflow
│   ├── run_predictions.py        # Generate weekly predictions
│   ├── ingest_fanfooty.py        # Scrape FanFooty data
│   ├── ingest_supercoach.py      # Scrape SuperCoach data
│   ├── build_features.py         # Parse and structure raw data
│   ├── train_model.py            # Train ML models
│   └── upload_to_gsheet.py       # Google Sheets integration
│
├── Jupyter Notebooks
│   ├── scrape-fanfooty.ipynb
│   ├── scrape-sc.ipynb
│   ├── predict-fanfooty.ipynb
│   ├── predict-sc.ipynb
│   ├── in-season.ipynb
│   └── supercoach_com_au_Scraper.ipynb
│
├── data/
│   ├── raw/                      # Raw scraped data
│   ├── processed/                # Cleaned and structured data
│   │   ├── master_fanfooty_data.csv
│   │   └── YYYY_round_N_fanfooty_data.csv
│   └── predictions/              # Generated predictions
│       └── YYYY_round_N_predictions.csv
│
├── models/                       # Trained ML models
│   ├── sc_regressor_model.joblib
│   ├── sc_classifier_model_gt80.joblib
│   └── sc_classifier_model_gt100.joblib
│
├── outputs/                      # League tracking outputs
│   ├── player_list.csv
│   ├── player_match_results.csv
│   ├── current_teams.csv
│   ├── fixture_results.csv
│   ├── ladder.csv
│   └── json files/
│
├── inputs/                       # Manual input files
│   ├── All Match Data/           # Historical match data
│   └── coach_list.csv
│
├── draft_prep/                   # Pre-season draft tools
│   └── SC 2025/
│       ├── supercoach-scrape-player-list.ipynb
│       └── 2025_SC_Player_list.csv
│
└── heat_map/                     # Draft heat map visualizations
    └── 2024/
```

## Data Sources

1. **FanFooty.com.au**
   - Historical match data and player statistics
   - Primary source for model training data

2. **fixturedownload.com**
   - AFL fixture information (JSON feed)
   - Used for upcoming match schedules

3. **SuperCoach.com.au**
   - Current season player data
   - League tracking information

## Models & Predictions

### Feature Set (22 features)

**Rolling Averages (9 features)**
- SuperCoach score, kicks, handballs, marks, tackles, hitouts, contested possessions, metres gained, time on ground

**Opponent Analysis (1 feature)**
- Opponent strength rating based on points conceded to position

**Player Roles (10 binary features)**
- Tagger, ruck, wing, job, star, hot, gun, shovel, guard, pocket

**Form & Durability (2 features)**
- Consistency over last 10 games
- Games played in previous season

### Model Architecture

All models use **RandomForestRegressor/Classifier** from scikit-learn:
- 200 trees
- min_samples_leaf=5
- max_features='sqrt'
- Evaluated with MAE (regression) and accuracy (classification)

### Prediction Output Format

CSV file with columns:
- Player ID
- Player name
- Team
- Position
- Opponent
- Projected SuperCoach score
- Probability of scoring 80+ points
- Probability of scoring 100+ points

## Configuration

### Google Sheets Setup

1. Create a Google Sheet named "SuperCoach Weekly Predictions"
2. Create a worksheet named "Waiver Targets"
3. Share the sheet with your service account email
4. Update `upload_to_gsheet.py` with your Sheet ID if different

### Customization

Edit configuration variables in individual scripts:
- Data years and rounds
- Model hyperparameters
- Feature selection
- Output file paths

## Weekly Workflow

1. **Friday/Saturday** (after round completes)
   ```bash
   python run_pipeline.py --round <completed_round>
   ```

2. **Review predictions in Google Sheets**
   - Check projected scores
   - Review matchup information
   - Identify high-probability performers

3. **Make team selection decisions**
   - Use predictions to inform captain choices
   - Identify trade targets
   - Set lineup for upcoming round

## Draft Preparation

Pre-season draft tools located in `draft_prep/SC 2025/`:

1. Run player scraping notebook to get current season data
2. Generate heat maps showing value picks by position
3. Review historical performance trends
4. Identify breakout candidates

## Contributing

This is a personal project, but suggestions and improvements are welcome. Feel free to fork and adapt for your own leagues.

## License

This project is for personal use. Please respect the terms of service of data sources (FanFooty, SuperCoach) when scraping.

## Acknowledgments

- FanFooty.com.au for comprehensive AFL statistics
- SuperCoach.com.au for fantasy league platform
- scikit-learn for machine learning tools

---

**Note**: This tool is designed to assist with fantasy football decisions but should not be the sole basis for team selection. Use in combination with injury news, team selection, weather conditions, and your own football knowledge.
