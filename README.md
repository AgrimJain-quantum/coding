# Coding Workspace

This repository is a personal coding workspace containing Python projects, C practice files, portfolio pages, datasets, and project outputs. It is organized by language or project type so each folder can be opened and run independently.

## Folder Overview

| Folder | Contents |
| --- | --- |
| `c/` | C language practice programs, including `chapter1.c` and `chapter2.c`, plus compiled `.exe` outputs. |
| `portfolios/` | Static portfolio HTML pages and image assets for a personal ML/AI engineer portfolio. |
| `python/` | Python learning exercises, games, utilities, data projects, and machine learning applications. |

## Repository Structure

```text
coding/
|-- c/
|-- portfolios/
|   |-- images/
|   `-- Agrim Jain — ML & AI Engineer_files/
|-- python/
|   |-- coffee machine/
|   |-- energy_simulator_app/
|   |-- machine learning model of electricity forecasting/
|   |-- Mail Merge Project Start/
|   |-- music recommendation project/
|   |-- nato names/
|   |-- ping pong game/
|   |-- python progress/
|   |-- quiz project/
|   |-- snake game/
|   |-- states game/
|   `-- turtle crossing/
`-- README.md
```

## Python Projects

| Project Folder | Description | Main Entry Point |
| --- | --- | --- |
| `python/coffee machine/` | Object-oriented console coffee machine simulation with menu, resource, and payment modules. | `main.py` |
| `python/energy_simulator_app/` | Energy data simulator experiments with multiple versions, example usage, and a static UI folder. | `run_simulator.py` |
| `python/machine learning model of electricity forecasting/` | Electricity load forecasting project with datasets, model versions, generated plots, and a modular V12 pipeline. | `main project/electricity_load_forecastingV12.py` |
| `python/Mail Merge Project Start/` | Mail merge script that fills a letter template with names and writes completed letters. | `main.py` |
| `python/music recommendation project/` | Streamlit music recommendation app using audio features and content-based similarity. | `app.py` |
| `python/nato names/` | NATO phonetic alphabet converter using a CSV lookup table. | `main.py` |
| `python/ping pong game/` | Turtle-based Pong game with paddle, ball, and scoreboard modules. | `main.py` |
| `python/python progress/` | Day-wise Python practice scripts, mini games, Tkinter apps, and course exercises. | Varies by file |
| `python/quiz project/` | Quiz application built with question, data, and quiz-brain modules. | `main.py` |
| `python/snake game/` | Turtle-based Snake game with food, snake, and scoreboard modules. | `main.py` |
| `python/states game/` | U.S. states guessing game using a map image and CSV state data. | `main.py` |
| `python/turtle crossing/` | Turtle arcade-style crossing game with player, car manager, and scoreboard modules. | `main.py` |

## Running Projects

Most Python projects can be run by opening the project folder and running the main file:

```bash
cd "python/snake game"
python main.py
```

For the music recommendation app:

```bash
cd "python/music recommendation project"
pip install -r requirements.txt
streamlit run app.py
```

For the electricity forecasting pipeline:

```bash
cd "python/machine learning model of electricity forecasting/main project"
python electricity_load_forecastingV12.py
```

For C practice files:

```bash
cd c
gcc chapter1.c -o chapter1
./chapter1
```

## Portfolio

The `portfolios/` folder contains static HTML files and image assets. Open the HTML files directly in a browser to view the portfolio pages.

## Notes

- Each project is mostly independent, so dependencies may differ by folder.
- Generated folders such as `__pycache__/`, compiled `.exe` files, generated letters, datasets, and plot outputs are present in the workspace.
- Check project-specific files such as `requirements.txt` or nested `README.md` files when available.
