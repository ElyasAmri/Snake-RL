# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an ECEN 446 course project implementing Reinforcement Learning applied to the classic Snake game. The project is part of Topic 1: "Reinforcement Learning and application to the snake game."

## Project Goals
1. Implement a playable Snake game environment
2. Develop a Reinforcement Learning agent that learns to play Snake
3. Demonstrate the trained agent
4. Produce a comprehensive report covering theory and implementation

## Deliverables
1. Full report (see REPORT-REQUIREMENTS.md)
2. Baseline models (deterministic)
2. Notebooks for training the models (one notebook for each model, for the purposes of the instructor) + a testing segment within the notebook that can be run independently of previous cells
3. A testing notebook that uses all models and produces all data and figures
4. Weights (normal + flood-fill)
5. Single snake game visualizer (training + testing) with algorithm selection
6. Double snake game visualizer (training + testing) with DQN only
7. Comprehensive README.md with documentation and navigation for the entire repo

## Structure
```
report/
results/
    weights/
    data/
    figures/
notebooks/
scripts/
    baselines/
    training/
    visualizer/
core/
tests/
```

## Running python scripts
```
# running a single script
./venv/Scripts/python.exe path/to/script

# running inline script
./venv/Scripts/python.exe -c "[script]"

# running tests
./venv/Scripts/python.exe -m pytest path/to/test
```

# Guidelines (IMPORTANT)
- Never use full path with Bash commands
- You are already in the project root. No need to use `cd` 
- Always test small segments before writing code to make sure they work as intended
- Never write big segments of code all at once. Test first (run sample in line code if you have to)
- Start small when writing code, then incremently add more logic
- Always use GPU for faster training (run N snakes at once)
- Never write files in root unless justified. Use appropriate folder
- Always keep everything organized
- Always search for theory for reinforcement learning agent before making suggestions
- Record metrics for training models, including time took to train
- When checking output using BashOutput, you must sleep for a period of time before checking again to avoid wasting tokens