#!/usr/bin/env python3
import argparse
import os
import pickle
import math
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import gymnasium as gym
import numpy as np
import pygame
from env.env_utils import read_point_distribution


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip01x512(v: float) -> float:
    return float(max(0.0, min(512.0, v)))


def play_demonstration(
    demo_path,
    env_name="gym_pusht/PushT-v0",
    obs_type="keypoints",
    workspace_size=512,
    save_images=False,
    screens_dir="screens_playback",
    fps=20,
):
    # Load the recorded list of step dicts
    with open(demo_path, "rb") as f:
        demo_data = pickle.load(f)

    # Single env in rgb_array mode (we'll display the frame ourselves)
    env = gym.make(
        env_name,
        obs_type=obs_type,
        render_mode="rgb_array",
        workspace_size=workspace_size,
    )

    # Add current .pkl file name to the path
    screens_dir = os.path.join(screens_dir, Path(demo_path).stem)

    if save_images:
        os.makedirs(screens_dir, exist_ok=True)

    # ---- Build correct 8D reset_to_state (agent, object, goal) ----
    if demo_data[0]["observation"] is not None:
        object_state = demo_data[0]["observation"]["object_state"]  # [obj_x, obj_y, obj_theta]
        agent_state = demo_data[0]["observation"]["agent_pos"]      # [agent_x, agent_y]
        goal_state = demo_data[0]["observation"]["goal_state"]      # [goal_x, goal_y, goal_theta]

        agent_x, agent_y = float(agent_state[0]), float(agent_state[1])
        obj_x, obj_y = float(object_state[0]), float(object_state[1])
        obj_theta = float(object_state[2])
        goal_x, goal_y = float(goal_state[0]), float(goal_state[1])
        goal_theta = float(goal_state[2])

        reset_state = [agent_x, agent_y, obj_x, obj_y, obj_theta, goal_x, goal_y, goal_theta]
        obs, info = env.reset(options={"reset_to_state": reset_state})
        print("reset to state", reset_state)
    else:
        obs, info = env.reset()
        print("reset to default random state")

    # ---- Setup pygame window to show frames in real time ----
    pygame.init()
    # Frame comes as HxWx3; window matches env workspace
    screen = pygame.display.set_mode((workspace_size*1.33, workspace_size*1.33))
    pygame.display.set_caption("PushT Playback (rgb_array)")
    clock = pygame.time.Clock()

    def blit_frame(frame: np.ndarray):
        # frame: HxWx3 uint8, need to transpose to WxHx3 for surfarray
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    def save_frame(step_idx: int, frame: np.ndarray):
        Image.fromarray(frame).save(os.path.join(screens_dir, f"{step_idx:06d}.png"))

    total_reward = 0.0

    # Initial render & show
    frame0 = env.render()
    if frame0 is None:
        raise RuntimeError("env.render() returned None; ensure render_mode='rgb_array'.")
    blit_frame(frame0)
    if save_images:
        save_frame(0, frame0)

    # ---- Playback loop ----
    for i in range(1, len(demo_data)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                pygame.quit()
                return

        # Recorded actions are absolute agent targets in pixels; clip to [0,512]
        action = np.array(demo_data[i]["action"], dtype=np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if obs_type == "keypoints":
            _ = read_point_distribution(next_obs)  # optional user utility

        # Get frame, display, maybe save
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None in rgb_array mode.")
        blit_frame(frame)
        if save_images:
            save_frame(i, frame)

        clock.tick(fps)
        if terminated or truncated:
            break

    print(f"Demonstration playback complete. Total reward: {total_reward}")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved demonstration")
    parser.add_argument("--demo_path", type=str, help="Path to the demonstration pickle file")
    parser.add_argument("--obs-type", type=str, default="keypoints", help="Observation type")
    parser.add_argument("--save-images", action="store_true", help="If set, save PNG frames while playing")
    parser.add_argument("--screens", type=str, default="database", help="Directory to save frames")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS")
    parser.add_argument("--workspace-size", type=int, default=512, help="Workspace size (pixels)")
    args = parser.parse_args()

    # Autopick the latest demo if not provided
    if args.demo_path is None:
        demo_folder = "database"
        demo_files = [f for f in os.listdir(demo_folder) if f.endswith(".pkl")]
        if not demo_files:
            raise FileNotFoundError("No demonstration files found in the 'demonstration' folder.")
        demo_files.sort()
        args.demo_path = os.path.join(demo_folder, demo_files[-1])
        print(f"No demo_path provided. Using last demo: {args.demo_path}")
    
    play_demonstration(
        demo_path=args.demo_path,
        obs_type=args.obs_type,
        workspace_size=args.workspace_size,
        save_images=args.save_images,
        screens_dir=args.screens,
        fps=args.fps,
    )
