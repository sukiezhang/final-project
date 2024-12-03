# BIOS611-final-project
Hi, this is my BIOS 611 final project. The project aims to train a model to predict an individual is at high risk or low risk of a heart attack 
based on the input parameters including patient’s socio-demographics, clinical  characteristics, lifestyle, and comorbidities.

Here is the project structure:
# Final Project Directory Structure
| Directory/File               | Description                           |
|------------------------------|---------------------------------------|
| `final_project/`             | Main project directory.              |
| `data/`                      | Contains datasets.                   |
| `data/original_data/`        | Raw datasets.                        |
| `train.csv`                  | Training dataset.                    |
| `test.csv`                   | Test dataset.                        |
| `sample_submission.csv`      | Example submission file.             |
| `output/`                    | Project outputs.                     |
| `output/figures/`            | Generated figures.                   |
| `output/tables/`             | Generated tables.                    |
| `final_report/`              | Final report files.                  |
| `report.tex`                 | LaTeX source file for the report.    |
| `bios611_final_report.pdf`   | Compiled report PDF.                 |
| `src/`                       | Source code.                         |
| `__init__.py`                | Initialization file.                 |
| `data_loader.py`             | Data loading script.                 |
| `data_processing.py`         | Data processing script.              |
| `data_exploration.py`        | Exploratory analysis script.         |
| `visualization.py`           | Visualization script.                |
| `modeling.py`                | Modeling script.                     |
| `bios611_final_project.ipynb`| Jupyter Notebook workflow.           |
| `Makefile`                   | Automation script.                   |
| `Dockerfile`                 | Docker configuration.                |
| `requirements.txt`           | Required Python packages.            |
| `main.py`                    | Main workflow script.                |

# Instructions on how to use the repo
First, build a docker image in the project directory with the Dockerfile. 
Second, run the Docker container and excute the Makefile to creat all the figures and tables saved in `output/`.
The main Python script is`main.py` and the souce codes are stored in `src/`. 
Third, use Latex to creat the final report with figures/tables inserted; you can find the Latex source file as `report.tex`. 
Final report compiled as `bios611_final_report.pdf` is saved under `final_project/`.
