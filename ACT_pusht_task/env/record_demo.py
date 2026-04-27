import argparse
import os
import pickle
import time
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import pygame


class DemonstrationRecorder:
    def __init__(
        self,
        env_name="gym_pusht/PushT-v0",
        obs_type="keypoints",
        input_device="keyboard",
        workspace_size=512,
        success_threshold=0.90,
        screenshot_dir="demonstration",
        fps=20,
        save_path=None,
    ):
        """
        Two synchronized envs:
          - env_display: render_mode='human'  -> shows a live window
          - env_capture: render_mode='rgb_array' -> returns frames to save
        """
        self.env_display = gym.make(
            env_name,
            obs_type=obs_type,
            render_mode="human",
            workspace_size=workspace_size,
            success_threshold=success_threshold,
        )
        self.env_capture = gym.make(
            env_name,
            obs_type=obs_type,
            render_mode="rgb_array",
            workspace_size=workspace_size,
            success_threshold=success_threshold,
        )

        self.input_device = input_device
        self.action_dim = self.env_display.action_space.shape[0]
        self.running = False
        self.current_episode = []
        self.demonstration = []
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.screenshot_dir = os.path.abspath(screenshot_dir)
        name = Path(save_path).stem if save_path else self.timestamp
        self.screenshot_dir = os.path.join(self.screenshot_dir, name)
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.fps = fps
        self._clock = None

    # ---------- Input handling ----------
    def setup_input(self):
        pygame.init()
        self.screen = pygame.display.set_mode((320, 240))
        pygame.display.set_caption("Demo Recorder (Keep this window focused!)")
        self._clock = pygame.time.Clock()

        if self.input_device == "joystick":
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Using joystick: {self.joystick.get_name()}")
            else:
                print("No joystick detected, falling back to keyboard")
                self.input_device = "keyboard"

    def get_keyboard_action(self):
        action = np.zeros(self.action_dim, dtype=np.float32)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        if keys[pygame.K_RIGHT]:
            action[0] = 1.0
        if keys[pygame.K_UP]:
            action[1] = -1.0
        if keys[pygame.K_DOWN]:
            action[1] = 1.0
        return action

    def get_joystick_action(self):
        action = np.zeros(self.action_dim, dtype=np.float32)
        if self.action_dim >= 2:
            action[0] = float(self.joystick.get_axis(0))
            action[1] = float(-self.joystick.get_axis(1))
        return action

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    # ---------- Frame saving ----------
    def save_current_frame(self, step_idx: int):
        """
        Save an image from env_capture (rgb_array) for this step.
        """
        frame = self.env_capture.render()
        if frame is None:
            print("[warn] env_capture.render() returned None; check render_mode='rgb_array'.")
            return
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"Unexpected frame shape {frame.shape}, expected HxWx3/4.")
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        img = Image.fromarray(frame)
        img.save(os.path.join(self.screenshot_dir, f"{step_idx:06d}.png"))

    # ---------- Episode saving ----------
    def save_episode(self):
        if len(self.current_episode) > 0:
            self.demonstration.append(self.current_episode)
            print(f"Saved episode with {len(self.current_episode)} steps")
            print("Total number of demonstrations:", len(self.demonstration))
            self.current_episode = []

    def save_demonstration(self, filepath=None):
        if not filepath:
            filepath = os.path.join(os.path.dirname(__file__), f"demonstration/{self.timestamp}.pkl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.current_episode, f)
        print(f"Demonstration with {len(self.demonstration)} episodes saved to {filepath}")

    # ---------- Recording loop ----------
    def record(self, max_steps=1000):
        self.setup_input()

        # Deterministic initial state shared by both envs
        rs = np.random.RandomState()
        state = np.array(
            [
                rs.randint(50, 450),
                rs.randint(50, 450),
                rs.randint(100, 400),
                rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi,
                300,
                300,
                np.pi / 4,
            ],
            dtype=np.float64,
        )

        obs_d, info_d = self.env_display.reset(options={"reset_to_state": state})
        obs_c, info_c = self.env_capture.reset(options={"reset_to_state": state})

        # Sanity: the two observations should be consistent (may differ in representation details)
        self.current_obs = obs_d

        print("\n===== Demonstration Recorder =====")
        print("Controls:")
        print("  Arrow keys / Joystick: Move the pusher")
        print("  ESC: Stop recording")
        print("  Live window: shown via env_display (human)")
        print("  PNGs saved each step via env_capture (rgb_array) -> ./screens")
        print("==================================\n")

        self.running = True
        self.step = 0
        pbar = tqdm(total=max_steps, desc="Recording")

        terminated = False
        truncated = False
        agent_pos = None

        while self.running and self.step < max_steps:
            # Show real-time sim
            _ = self.env_display.render()

            self.process_events()

            # Action from input device
            if self.input_device == "keyboard":
                delta = self.get_keyboard_action()
            else:
                delta = self.get_joystick_action()

            # PushT usually wants absolute agent pos; derive from obs/get_obs if available
            if self.env_display.get_obs() is not None:
                agent_pos = self.env_display.get_obs().get("agent_pos", None)
            if agent_pos is None:
                action = delta * 10.0
            else:
                action = agent_pos + delta * 10.0            
            if self.env_display.get_obs() is None:        
                agent_pos = action
                
            # Step BOTH envs with the SAME action
            next_obs_d, reward_d, term_d, trunc_d, info_d = self.env_display.step(action)
            next_obs_c, reward_c, term_c, trunc_c, info_c = self.env_capture.step(action)

            # Save frame from capture env every step
            self.save_current_frame(self.step)

            # Record transition (use display obs for consistency with what you see)
            step_data = {
                "observation": self.current_obs.tolist() if hasattr(self.current_obs, "tolist") else self.current_obs,
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": float(reward_d),
                # "image": np.array(self.env_capture.render(), dtype=np.uint8),  # HxWx3
                "next_observation": next_obs_d.tolist() if hasattr(next_obs_d, "tolist") else next_obs_d,
            }

            self.current_episode.append(step_data)
            self.current_obs = next_obs_d

            # Consistency checks
            if (term_d != term_c) or (trunc_d != trunc_c):
                print("[warn] display/capture env diverged on done flags. Continuing, but check determinism.")
            if reward_d != reward_c:
                # tiny float noise is okay; this is a hard check, so only print when obviously different
                if abs(reward_d - reward_c) > 1e-6:
                    print("[warn] display/capture env rewards differ. Check env determinism.")

            terminated = term_d
            truncated = trunc_d

            pbar.update(1)
            pbar.set_postfix({"episode": len(self.demonstration) + 1})
            self.step += 1

            if terminated or truncated:
                pbar.close()
                if terminated:
                    print("Congratulations, task completed successfully!")
                if truncated:
                    print("Episode truncated (time limit or other condition).")
                break

            self._clock.tick(self.fps)

        pbar.close()
        self.env_display.close()
        self.env_capture.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Record demonstrations for gym-pusht")
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "joystick"], help="Input device")
    parser.add_argument("--output", type=str, default=None, help="Output file path for pickle")
    parser.add_argument("--obs-type", type=str, default="keypoints", help="Observation type")
    parser.add_argument("--fps", type=int, default=20, help="Control loop FPS")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps to run")
    args = parser.parse_args()

    recorder = DemonstrationRecorder(
        env_name="gym_pusht/PushT-v0",
        obs_type=args.obs_type,
        input_device=args.device,
        fps=args.fps,
        save_path=args.output,
    )

    recorder.record(max_steps=args.max_steps)
    recorder.save_demonstration(args.output)


if __name__ == "__main__":
    main()