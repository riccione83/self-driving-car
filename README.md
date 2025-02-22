# Autonomous Driving Simulation

This is a weekend project I built in just 5 hours to explore autonomous driving through deep learning. Using Python, Pygame, and TensorFlow, this simulation trains a virtual car to navigate a randomly generated track with a Deep Q-Network (DQN). The car learns from both autonomous exploration and manual driving inputs, guided by Grok 3 (xAI’s AI assistant) throughout the development process.

## Features

- **Deep Learning**: A DQN (256-128-64 layers) trained with TensorFlow to optimize the car’s navigation policy.
- **Autonomous Mode**: The car explores the track using an epsilon-greedy strategy (epsilon: 1.0 to 0.01), balancing exploration and learned actions.
- **Manual Override**: Take control with arrow keys to demonstrate optimal paths, with boosted rewards (2x) integrated into the AI’s memory.
- **Track Generation**: Random tracks created with `scipy.interpolate.splprep`, though I had to manually tweak some for smoothness.
- **Visual Feedback**: The car turns green on good trajectories, red when off-track, and features Tesla-inspired steering wheel indicators.
- **Lap Counting**: Counts laps when crossing the start/finish line after ~80% track completion, with a cooldown to prevent duplicates.

## How It Works

The car uses sensors (9 rays from -90° to 90°) to detect track boundaries, feeding data into the DQN alongside speed, angle, and track progress. Actions include accelerate, decelerate, turn left, and turn right. Rewards encourage staying on track (+1), making progress (+500 scaled), and completing laps (+500), with penalties for going off-track (-100) or getting stuck (-2).

- **Deep Learning**: The DQN trains on batches of 64 experiences, stored in a 20,000-entry replay memory, with frequent target network updates (every 5 episodes).
- **Manual Learning**: Manual inputs are amplified and stored, speeding up convergence via imitation learning within the reinforcement framework.

## Installation

1. **Clone the Repo**:
   Requirement: Python 3.10 or newer
   ```bash
   git clone https://github.com/riccione83/self-driving-car.git
   cd autonomous-driving-sim
   pip install pygame numpy tensorflow scipy
   python car.py
   ```

## Notes

- **Virtual Environment (Recommended)**: To avoid conflicts with other projects:
  ```bash
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  venv\Scripts\activate     # Windows
  pip install pygame numpy tensorflow scipy
  ```
