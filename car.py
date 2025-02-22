import pygame
import numpy as np
import tensorflow as tf
from collections import deque
import random
import math
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy.interpolate import splprep, splev

# Global parameters
FAST_LEARNING = False
FIXED_START = True
FIXED_TRACK = False
EMERGENCY_CORRECTION = True
EMERGENCY_THRESHOLD = 0.15
STUCK_THRESHOLD = 0.1
STUCK_TIME_LIMIT = 180
STUCK_PENALTY = -2.0
DEATH_PENALTY = -100
SURVIVAL_REWARD = 1.0
TIME_PENALTY = -0.02
MANUAL_OVERRIDE_REWARD = 20.0
DEEP_TRAIN_EPISODES = 100
BATCH_SIZE = 64
PARALLEL_EPISODES = 8   # Adjust as needed
USE_PARALLEL_GPU = False
MANUAL_BOOST_FACTOR = 2.0
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
GOOD_PROGRESS_THRESHOLD = 0.01

# Steering wheel indicator settings
INDICATOR_SIZE = 10
INDICATOR_OFFSET_LEFT = (-20, 10)
INDICATOR_OFFSET_RIGHT = (20, 10)
INDICATOR_COLOR_OFF = (100, 100, 100)
INDICATOR_COLOR_ON = (255, 215, 0)

# Constants
WIDTH, HEIGHT = 1024, 760
CAR_SIZE = 20
FPS = 60
TRACK_WIDTH = 80
PRE_TRAIN_EPISODES = 100
MODEL_PATH = "car_model.weights.h5"
MANUAL_STEPS = 5000

# Reward scaling constants
PROGRESS_SCALE = 20.0
CENTERING_SCALE = 2.0
SENSOR_PENALTY_SCALE = 5.0
RACING_LINE_SCALE = 10.0
LAP_REWARD_BASE = 500
OPTIMAL_LAP_TIME = 1000

# Colors
WHITE = (245, 245, 245)
GRAY = (100, 100, 100)
BLACK = (20, 20, 20)
RED = (255, 80, 80)
BLUE = (80, 80, 255)
GREEN = (0, 255, 0)
GOOD_COLOR = GREEN
BAD_COLOR = RED

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    if USE_PARALLEL_GPU and len(physical_devices) > 1:
        print(f"Using {len(physical_devices)} GPUs with MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print(f"Using {len(physical_devices)} GPU(s) with default strategy")
        strategy = tf.distribute.get_strategy()
else:
    print("Using CPU for training")
    strategy = tf.distribute.get_strategy()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car - Survival Training")
clock = pygame.time.Clock()

def generate_random_track():
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    base_radius = min(WIDTH, HEIGHT) // 3
    num_points = 10
    control_points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        offset = random.uniform(-base_radius * 0.3, base_radius * 0.3)
        r = base_radius + offset
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        control_points.append((x, y))
    control_points.append(control_points[0])
    pts = np.array(control_points).T
    tck, u = splprep(pts, s=0, per=True)
    u_fine = np.linspace(0, 1, 200)
    x_fine, y_fine = splev(u_fine, tck)
    return [(int(x), int(y)) for x, y in zip(x_fine, y_fine)]

def get_centerline(track):
    return [((track[i][0] + track[i+1][0])/2, (track[i][1] + track[i+1][1])/2) 
            for i in range(len(track)-1)]

def draw_track(screen, track):
    pygame.draw.polygon(screen, GRAY, track, TRACK_WIDTH)
    pygame.draw.polygon(screen, BLACK, track, 2)
    pygame.draw.line(screen, GREEN, track[0], track[1], 5)  # Start/finish line

def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return False
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    return 0 <= t <= 1 and 0 <= u <= 1

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)

class Car:
    def __init__(self, track):
        self.track = track
        self.centerline = get_centerline(track)
        self.sensor_angles = [-90, -60, -30, -15, 0, 15, 30, 60, 90]
        self.sensor_distances = [0] * len(self.sensor_angles)
        self.max_speed = 2.0
        self.stuck_counter = 0
        self.lap_progress = 0
        self.laps = 0
        self.last_crossed_time = -1  # Timestamp of last crossing
        self.cooldown = 60  # Frames to wait before counting another lap
        self.left_indicator = False
        self.right_indicator = False
        self.last_position = None
        self.progress = 0.0
        self.reset()

    def reset(self):
        p1, p2 = self.track[0], self.track[1]
        self.x = (p1[0] + p2[0]) / 2 if FIXED_START else random.uniform(0, WIDTH - CAR_SIZE)
        self.y = (p1[1] + p2[1]) / 2 if FIXED_START else random.uniform(0, HEIGHT - CAR_SIZE)
        self.angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) if FIXED_START else random.uniform(0, 360)
        self.speed = 0.5
        self.stuck_counter = 0
        self.prev_x = self.x
        self.prev_y = self.y
        self.lap_progress = 0
        self.lap_frames = 0
        self.left_indicator = False
        self.right_indicator = False
        self.last_position = (self.x, self.y)
        self.progress = 0.0
        self.last_crossed_time = -1

    def move(self, action):
        if action == 0:
            self.speed += 0.15
        elif action == 1:
            self.speed -= 0.1
            if self.speed < 0:
                self.speed = 0
        elif action == 2:
            self.angle += 6
            self.left_indicator = True
            self.right_indicator = False
        elif action == 3:
            self.angle -= 6
            self.left_indicator = False
            self.right_indicator = True
        else:
            self.left_indicator = False
            self.right_indicator = False
        
        self.speed = max(0, min(self.max_speed, self.speed))
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))
        self.x = max(0, min(WIDTH - CAR_SIZE, self.x))
        self.y = max(0, min(HEIGHT - CAR_SIZE, self.y))
        
        distance_moved = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
        self.prev_x = self.x
        self.prev_y = self.y
        
        if self.speed < STUCK_THRESHOLD and distance_moved < STUCK_THRESHOLD:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.lap_frames += 1
        self.update_progress()
        return self.stuck_counter >= STUCK_TIME_LIMIT

    def update_progress(self):
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(self.centerline):
            dist = math.hypot(self.x - point[0], self.y - point[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        self.progress = closest_idx / len(self.centerline)
        # Update lap_progress to track maximum progress
        if self.progress > self.lap_progress:
            self.lap_progress = self.progress

    def check_lap_completion(self, prev_x, prev_y):
        x1, y1 = self.track[0]  # Start point
        x2, y2 = self.track[1]  # Finish point
        crossed = line_intersection(prev_x, prev_y, self.x, self.y, x1, y1, x2, y2)
        
        # print(f"Checking lap: Progress={self.lap_progress:.2f}, "
        #       f"Position=({self.x:.1f}, {self.y:.1f}), "
        #       f"Prev=({prev_x:.1f}, {prev_y:.1f}), "
        #       f"Line=({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f}), "
        #       f"Crossed={crossed}")
        
        if crossed and self.lap_progress > 0.8 and (self.lap_frames - self.last_crossed_time) > self.cooldown:
            print(f"Lap detected! Laps={self.laps + 1}, Frames={self.lap_frames}")
            self.laps += 1
            self.lap_progress = 0
            self.progress = 0
            self.last_crossed_time = self.lap_frames
            return True
        return False

    def update_sensors(self):
        for i, angle in enumerate(self.sensor_angles):
            sensor_angle = self.angle + angle
            self.sensor_distances[i] = self.cast_ray(sensor_angle)

    def cast_ray(self, angle):
        ray_length = 0
        max_range = 200
        while ray_length < max_range:
            ray_x = self.x + ray_length * math.cos(math.radians(angle))
            ray_y = self.y + ray_length * math.sin(math.radians(angle))
            if not self.is_point_inside_polygon(ray_x, ray_y):
                return ray_length / max_range
            ray_length += 2
        return 1.0

    def is_point_inside_polygon(self, x, y):
        inside = False
        for i in range(len(self.track) - 1):
            x1, y1 = self.track[i]
            x2, y2 = self.track[i + 1]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside

    def is_on_track(self):
        min_dist = float('inf')
        for i in range(len(self.track) - 1):
            x1, y1 = self.track[i]
            x2, y2 = self.track[i + 1]
            d = point_to_segment_distance(self.x, self.y, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
        return min_dist <= TRACK_WIDTH / 2

    def get_racing_line_distance(self):
        min_dist = float('inf')
        for point in self.centerline:
            dist = math.hypot(self.x - point[0], self.y - point[1])
            min_dist = min(min_dist, dist)
        return min_dist

    def is_good_trajectory(self, prev_progress):
        return (self.progress - prev_progress) > GOOD_PROGRESS_THRESHOLD and self.is_on_track()

    def draw(self, death_flash=False, good_trajectory=False):
        points = [
            (self.x + CAR_SIZE * math.cos(math.radians(self.angle)),
             self.y + CAR_SIZE * math.sin(math.radians(self.angle))),
            (self.x + CAR_SIZE/2 * math.cos(math.radians(self.angle + 135)),
             self.y + CAR_SIZE/2 * math.sin(math.radians(self.angle + 135))),
            (self.x + CAR_SIZE/2 * math.cos(math.radians(self.angle - 135)),
             self.y + CAR_SIZE/2 * math.sin(math.radians(self.angle - 135)))
        ]
        color = BLACK if death_flash else (GOOD_COLOR if good_trajectory else BAD_COLOR)
        pygame.draw.polygon(screen, color, points)
        
        left_pos = (self.x + INDICATOR_OFFSET_LEFT[0], self.y + INDICATOR_OFFSET_LEFT[1])
        right_pos = (self.x + INDICATOR_OFFSET_RIGHT[0], self.y + INDICATOR_OFFSET_RIGHT[1])
        left_color = INDICATOR_COLOR_ON if self.left_indicator else INDICATOR_COLOR_OFF
        right_color = INDICATOR_COLOR_ON if self.right_indicator else INDICATOR_COLOR_OFF
        
        pygame.draw.circle(screen, left_color, left_pos, INDICATOR_SIZE)
        pygame.draw.circle(screen, right_color, right_pos, INDICATOR_SIZE)
        
        for i, (angle, dist) in enumerate(zip(self.sensor_angles, self.sensor_distances)):
            end_x = self.x + dist * 200 * math.cos(math.radians(self.angle + angle))
            end_y = self.y + dist * 200 * math.sin(math.radians(self.angle + angle))
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)

class DQNAgent:
    def __init__(self, state_size, action_size, load_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = EPSILON_START if not load_model else EPSILON_MIN
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = 0.001
        with strategy.scope():
            self.model = self._build_model()
            self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = BATCH_SIZE
        self.lock = threading.Lock()

        if load_model and os.path.exists(MODEL_PATH):
            self.model.load_weights(MODEL_PATH)
            self.update_target_model()
            print(f"Loaded model from {MODEL_PATH}")

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def train(self, batch_data):
        with self.lock:
            for state, action, reward, next_state, done in batch_data:
                self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states = np.vstack([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.vstack([x[3] for x in batch])
            dones = np.array([x[4] for x in batch])
            targets = rewards + (1 - dones) * self.gamma * np.max(self.target_model.predict(next_states, verbose=0), axis=1)
            target_f = self.model.predict(states, verbose=0)
            for i, action in enumerate(actions):
                target_f[i][action] = targets[i]
            self.model.fit(states, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def train_from_demonstration(self, states, actions, rewards):
        states = np.vstack(states)
        targets = self.model.predict(states, verbose=0)
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            targets[i][action] = reward * MANUAL_BOOST_FACTOR
        self.model.fit(states, targets, epochs=20, verbose=1)
        self.memory.extend([(s, a, r * MANUAL_BOOST_FACTOR, s, False) 
                          for s, a, r in zip(states, actions, rewards)])
        print("Completed demonstration training")

    def save_model(self):
        self.model.save_weights(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

def run_episode(agent, track, start_line_point):
    car = Car(track)
    car.reset()
    episode_data = []
    total_reward = 0
    steps = 0
    max_steps = 2000
    
    state = np.array(car.sensor_distances + [car.speed / car.max_speed, 
                                            car.angle / 360, 
                                            car.progress], 
                    dtype=np.float32)
    state = np.reshape(state, [1, agent.state_size])
    
    while steps < max_steps:
        car.update_sensors()
        action = agent.act(state)
        prev_x, prev_y, prev_angle = car.x, car.y, car.angle
        prev_progress = car.progress
        is_stuck = car.move(action)
        car.update_sensors()
        
        reward = SURVIVAL_REWARD if car.is_on_track() else DEATH_PENALTY
        progress_reward = (car.progress - prev_progress) * PROGRESS_SCALE * 100
        left_avg = np.mean(car.sensor_distances[0:4])
        right_avg = np.mean(car.sensor_distances[5:])
        centering_reward = (1 - abs(left_avg - right_avg)) * CENTERING_SCALE
        sensor_penalty = sum(max(0, 0.15 - s) for s in car.sensor_distances) * SENSOR_PENALTY_SCALE
        speed_reward = car.speed * 0.5
        racing_line_reward = (1 - car.get_racing_line_distance() / (TRACK_WIDTH / 2)) * RACING_LINE_SCALE
        reward += progress_reward + centering_reward + speed_reward - sensor_penalty + racing_line_reward + TIME_PENALTY
        
        if car.speed < STUCK_THRESHOLD:
            reward += STUCK_PENALTY
        
        if car.check_lap_completion(prev_x, prev_y):
            lap_time_bonus = max(0, (OPTIMAL_LAP_TIME - car.lap_frames) / OPTIMAL_LAP_TIME * 150)
            reward += LAP_REWARD_BASE + lap_time_bonus
            car.lap_frames = 0
        
        done = is_stuck or not car.is_on_track()
        if done:
            reward = DEATH_PENALTY
        
        total_reward += reward
        next_state = np.array(car.sensor_distances + [car.speed / car.max_speed, 
                                                    car.angle / 360, 
                                                    car.progress], 
                            dtype=np.float32)
        next_state = np.reshape(next_state, [1, agent.state_size])
        episode_data.append((state, action, reward, next_state, done))
        state = next_state
        steps += 1
        if done:
            break
    
    return episode_data, total_reward, car.laps

def deep_train(agent, car, episodes, start_line_point):
    print(f"Starting deep training for {episodes} episodes with {PARALLEL_EPISODES} threads...")
    total_laps = 0
    total_reward_sum = 0
    all_episode_data = []
    
    with ThreadPoolExecutor(max_workers=PARALLEL_EPISODES) as executor:
        batch_size = PARALLEL_EPISODES
        for batch_start in range(0, episodes, batch_size):
            batch_end = min(batch_start + batch_size, episodes)
            futures = [executor.submit(run_episode, agent, car.track, start_line_point) 
                      for _ in range(batch_start, batch_end)]
            
            batch_data = []
            for future in futures:
                episode_data, reward, laps = future.result()
                batch_data.extend(episode_data)
                total_reward_sum += reward
                total_laps += laps
            
            all_episode_data.extend(batch_data)
            
            if len(all_episode_data) >= BATCH_SIZE:
                agent.train(all_episode_data[:BATCH_SIZE])
                all_episode_data = all_episode_data[BATCH_SIZE:]
            
            if (batch_start // batch_size + 1) % 5 == 0:
                avg_reward = total_reward_sum / (batch_start + batch_size)
                print(f"Deep Train Batch {batch_start // batch_size + 1}, "
                      f"Total Laps: {total_laps}, Avg Reward: {avg_reward:.1f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
    
    if all_episode_data:
        agent.train(all_episode_data)
    
    avg_reward = total_reward_sum / episodes if episodes > 0 else 0
    print(f"Deep training completed. Total Laps: {total_laps}, Avg Reward: {avg_reward:.1f}")
    agent.save_model()

def manual_drive(car, agent):
    print("Manual driving mode: Use arrow keys to drive (UP: accelerate, DOWN: slow down, LEFT/RIGHT: turn)")
    print(f"Recording {MANUAL_STEPS} steps. Press 'Q' to finish early.")
    states, actions, rewards = [], [], []
    step = 0
    lap_count = 0
    start_line_point = ((car.track[0][0] + car.track[1][0]) / 2, 
                       (car.track[0][1] + car.track[1][1]) / 2)
    running = True
    death_flash = False
    
    while running and step < MANUAL_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]: action = 0
        elif keys[pygame.K_DOWN]: action = 1
        elif keys[pygame.K_LEFT]: action = 2
        elif keys[pygame.K_RIGHT]: action = 3
        
        car.update_sensors()
        state = np.array(car.sensor_distances + [car.speed / car.max_speed, 
                                               car.angle / 360, 
                                               car.progress], 
                        dtype=np.float32)
        state = np.reshape(state, [1, agent.state_size])
        
        prev_x, prev_y = car.x, car.y
        is_stuck = car.move(action)
        
        reward = SURVIVAL_REWARD if car.is_on_track() else DEATH_PENALTY
        if car.check_lap_completion(prev_x, prev_y):
            lap_count = car.laps
            reward += LAP_REWARD_BASE
            car.lap_frames = 0
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        death_flash = False
        if is_stuck or not car.is_on_track():
            car.reset()
            death_flash = True
        
        screen.fill(WHITE)
        draw_track(screen, car.track)
        car.draw(death_flash=death_flash, good_trajectory=car.is_good_trajectory(car.progress - 0.01))
        text = pygame.font.Font(None, 36).render(f"Manual Steps: {step}/{MANUAL_STEPS}  Laps: {lap_count}", True, BLACK)
        screen.blit(text, (10, 10))
        status = "Alive" if car.is_on_track() else "Dead"
        status_text = pygame.font.Font(None, 36).render(f"Status: {status}", True, BLACK)
        screen.blit(status_text, (10, 50))
        pygame.display.flip()
        clock.tick(FPS)
        step += 1
    
    return states, actions, rewards

def main():
    track = generate_random_track()
    car = Car(track)
    agent = DQNAgent(state_size=len(car.sensor_angles) + 3, action_size=4, load_model=True)
    font = pygame.font.Font(None, 36)
    
    start_line_point = ((track[0][0] + track[1][0]) / 2, (track[0][1] + track[1][1]) / 2)
    lap_count = 0
    last_lap_time = None
    local_buffer = []
    death_flash = False
    
    if not os.path.exists(MODEL_PATH):
        print("No saved model found. Starting manual training...")
        states, actions, rewards = manual_drive(car, agent)
        agent.train_from_demonstration(states, actions, rewards)
        agent.save_model()
    else:
        print("Using pre-trained model")
    
    episode = 0
    total_reward = 0
    manual_override_mode = False
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                agent.save_model()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    track = generate_random_track()
                    car = Car(track)
                    start_line_point = ((track[0][0] + track[1][0]) / 2, 
                                      (track[0][1] + track[1][1]) / 2)
                    episode = 0
                    total_reward = 0
                    lap_count = 0
                    last_lap_time = None
                    local_buffer = []
                elif event.key == pygame.K_l:
                    deep_train(agent, car, DEEP_TRAIN_EPISODES, start_line_point)                
                elif event.key == pygame.K_s:
                    agent.save_model()
                elif event.key == pygame.K_SPACE:
                    manual_override_mode = not manual_override_mode
                    print(f"Manual Override {'Enabled' if manual_override_mode else 'Disabled'}")
        
        car.update_sensors()
        state = np.array(car.sensor_distances + [car.speed / car.max_speed, 
                                               car.angle / 360, 
                                               car.progress], 
                        dtype=np.float32)
        state = np.reshape(state, [1, agent.state_size])
        
        prev_x, prev_y, prev_angle = car.x, car.y, car.angle
        prev_progress = car.progress
        
        if manual_override_mode:
            keys = pygame.key.get_pressed()
            action = 0
            if keys[pygame.K_UP]: action = 0
            elif keys[pygame.K_DOWN]: action = 1
            elif keys[pygame.K_LEFT]: action = 2
            elif keys[pygame.K_RIGHT]: action = 3
        else:
            action = agent.act(state)
            if EMERGENCY_CORRECTION:
                left_min = min(car.sensor_distances[0:4])
                right_min = min(car.sensor_distances[5:])
                if left_min < EMERGENCY_THRESHOLD or right_min < EMERGENCY_THRESHOLD:
                    diff = left_min - right_min
                    action = 2 if diff < -0.05 else 3 if diff > 0.05 else action
        
        is_stuck = car.move(action)
        car.update_sensors()
        
        reward = SURVIVAL_REWARD if car.is_on_track() else DEATH_PENALTY
        progress_reward = (car.progress - prev_progress) * PROGRESS_SCALE * 100
        left_avg = np.mean(car.sensor_distances[0:4])
        right_avg = np.mean(car.sensor_distances[5:])
        centering_reward = (1 - abs(left_avg - right_avg)) * CENTERING_SCALE
        sensor_penalty = sum(max(0, 0.15 - s) for s in car.sensor_distances) * SENSOR_PENALTY_SCALE
        speed_reward = car.speed * 0.5
        racing_line_reward = (1 - car.get_racing_line_distance() / (TRACK_WIDTH / 2)) * RACING_LINE_SCALE
        reward += progress_reward + centering_reward + speed_reward - sensor_penalty + racing_line_reward + TIME_PENALTY
        
        if car.speed < STUCK_THRESHOLD:
            reward += STUCK_PENALTY
        if manual_override_mode:
            reward += MANUAL_OVERRIDE_REWARD
        
        if car.check_lap_completion(prev_x, prev_y):
            lap_time_bonus = max(0, (OPTIMAL_LAP_TIME - car.lap_frames) / OPTIMAL_LAP_TIME * 150)
            reward += LAP_REWARD_BASE + lap_time_bonus
            lap_count = car.laps
            last_lap_time = car.lap_frames
            print(f"Lap {lap_count} completed in {last_lap_time} frames. Bonus: {lap_time_bonus:.1f}")
            car.lap_frames = 0
        
        done = False
        death_flash = False
        if is_stuck or not car.is_on_track():
            reward = DEATH_PENALTY
            car.reset()
            episode += 1
            done = True
            death_flash = True
            if is_stuck:
                print(f"Car was stuck, resetting. Episode: {episode}")
            else:
                print(f"Driver died (off-track), resetting. Episode: {episode}")
        
        total_reward += reward
        next_state = np.array(car.sensor_distances + [car.speed / car.max_speed, 
                                                    car.angle / 360, 
                                                    car.progress], 
                            dtype=np.float32)
        next_state = np.reshape(next_state, [1, agent.state_size])
        
        if manual_override_mode:
            reward *= MANUAL_BOOST_FACTOR
        local_buffer.append((state, action, reward, next_state, done))
        
        if len(local_buffer) >= BATCH_SIZE:
            agent.train(local_buffer)
            local_buffer = []
        
        if manual_override_mode:
            print(f"Manual Action: {action}, Reward: {reward:.1f}")
        
        if episode % 5 == 0 and episode > 0:
            agent.update_target_model()
        
        if not FAST_LEARNING:
            screen.fill(WHITE)
            draw_track(screen, track)
            car.draw(death_flash=death_flash, good_trajectory=car.is_good_trajectory(prev_progress))
            text = font.render(f"Episode: {episode}  Reward: {total_reward:.1f}  Epsilon: {agent.epsilon:.3f}", True, BLACK)
            screen.blit(text, (10, 10))
            save_text = font.render("Press 'S' to save, 'R' for new track, 'SPACE' for manual, 'L' for deep learning", True, BLACK)
            screen.blit(save_text, (10, 40))
            status = "Alive" if car.is_on_track() else "Dead"
            status_text = font.render(f"Status: {status}", True, BLACK)
            screen.blit(status_text, (10, 80))
            lap_text = font.render(f"Laps: {lap_count}", True, BLACK)
            screen.blit(lap_text, (10, 110))
            if last_lap_time:
                time_text = font.render(f"Last Lap: {last_lap_time} frames", True, BLACK)
                screen.blit(time_text, (10, 140))
            if car.stuck_counter > 0:
                stuck_text = font.render(f"Stuck Counter: {car.stuck_counter}/{STUCK_TIME_LIMIT}", True, RED)
                screen.blit(stuck_text, (10, 170))
            mode_text = font.render(f"Mode: {'Manual' if manual_override_mode else 'Auto'}", 
                                  True, GREEN if manual_override_mode else BLACK)
            screen.blit(mode_text, (10, 200))
            pygame.display.flip()
            clock.tick(FPS)
    
    if local_buffer:
        agent.train(local_buffer)
    pygame.quit()

def train_from_demonstration(self, states, actions, rewards):
    states = np.vstack(states)
    targets = np.zeros((len(actions), self.action_size))
    for i, action in enumerate(actions):
        targets[i][action] = rewards[i] * MANUAL_BOOST_FACTOR
    self.model.fit(states, targets, epochs=20, verbose=1)
    self.memory.extend([(s, a, r * MANUAL_BOOST_FACTOR, s, False) 
                       for s, a, r in zip(states, actions, rewards)])
    print("Completed demonstration training")

DQNAgent.train_from_demonstration = train_from_demonstration

if __name__ == "__main__":
    main()