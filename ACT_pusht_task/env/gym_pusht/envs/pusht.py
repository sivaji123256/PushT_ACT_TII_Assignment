import collections
import os
import warnings

import cv2
import gymnasium as gym
import numpy as np

with warnings.catch_warnings():
    # Filter out DeprecationWarnings raised from pkg_resources
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning)
    import pygame

import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from .pymunk_override import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    """
    ## Description

    PushT environment.

    The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

    ## Action Space

    The action space is continuous and consists of two values: [x, y]. The values are in the range [0, workspace_size] and
    represent the target position of the agent.

    ## Observation Space

    If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
    environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, workspace_size] for the agent
    and block positions and [0, 2*pi] for the block angle.

    If `obs_type` is set to `keypoints` the observation space is a dictionary with:
    - `environment_state`: 16-dimensional vector representing the keypoint locations of the T (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, workspace_size]. See `get_keypoints` for a diagram showing the location of the
        keypoint indices.
    - `agent_pos`: A 2-dimensional vector representing the position of the robot end-effector.

    If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

    ## Rewards

    The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

    ## Success Criteria

    The environment is considered solved if the block is at least 95% in the goal zone.

    ## Starting State

    The agent starts at a random position and the block starts at a random position and angle.

    ## Episode Termination

    The episode terminates when the block is at least 95% in the goal zone.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
    ```

    * `obs_type`: (str) The observation type. Can be either `state`, `keypoints`, `pixels` or `pixels_agent_pos`.
      Default is `state`.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `observation_width`: (int) The width of the observed image. Default is `96`.

    * `observation_height`: (int) The height of the observed image. Default is `96`.

    * `visualization_width`: (int) The width of the visualized image. Default is `680`.

    * `visualization_height`: (int) The height of the visualized image. Default is `680`.

    * `workspace_size`: (int) The size of the square workspace in pixels. Default is `512`.

    ## Reset Arguments

    Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

    > [!WARNING]
    > For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
    > the one passed in the argument.

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0")
    >>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
    >>> state
    array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
          dtype=float32)
    ```

    ## Version History

    * v0: Original version

    ## References

    * TODO:
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        render_contact=False,
        block_cog=None,
        damping=None,
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
        workspace_size=512,
        success_threshold=0.85
    ):
        super().__init__()
        # Observations
        self.obs_type = obs_type
        self.render_contact = render_contact 
        # Workspace dimensions
        self.workspace_size = workspace_size

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Initialize spaces
        self._initialize_observation_space()
        self.action_space = spaces.Box(low=0, high=self.workspace_size, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        self.wall_thickness = 5  # Thicker walls for better collision detection

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = success_threshold  # 95% coverage
        
        # Initialize pressure data
        self.reset_pressure_data()

    def reset_pressure_data(self):
        """Reset pressure data for new episode."""
        self.contact_forces = []
        self.contact_positions = []
        
    def get_contact_data(self):
        """Get raw contact forces and positions."""
        return {
            'forces': np.array(self.contact_forces) if self.contact_forces else np.array([]),
            'positions': np.array(self.contact_positions) - np.array(self.agent.position) if self.contact_positions else np.array([]).reshape(0, 2)
        }
    

    def _initialize_observation_space(self):
        if self.obs_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0]),
                high=np.array([self.workspace_size, self.workspace_size, self.workspace_size, self.workspace_size, 2 * np.pi]),
                dtype=np.float64,
            )
        elif self.obs_type == "keypoints":
            self.observation_space = spaces.Dict(
                {   
                    "object_state": spaces.Box(
                        low=np.array([0, 0, 0]),
                        high=np.array([self.workspace_size, self.workspace_size, 2 * np.pi]),
                        dtype=np.float64,
                    ),
                    "goal_state": spaces.Box(
                        low=np.array([0, 0, 0]),
                        high=np.array([self.workspace_size, self.workspace_size, 2 * np.pi]),
                        dtype=np.float64,
                    ),
                    "object_keypoints": spaces.Box(
                        low=np.zeros(16),
                        high=np.full((16,), self.workspace_size),
                        dtype=np.float64,
                    ),
                    "goal_keypoints": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([self.workspace_size, self.workspace_size]),
                        dtype=np.float64,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([self.workspace_size, self.workspace_size]),
                        dtype=np.float64,
                    ),
                },
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([self.workspace_size, self.workspace_size]),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, keypoints, "
                "pixels_agent_pos]"
            )

    def _get_coverage(self):
        goal_body = self.get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)

        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return intersection_area / goal_area

    def step(self, action):
        # Reset pressure data for this step
        self.reset_pressure_data()
        
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)
            
            # Check and correct wall penetration
            self._prevent_wall_penetration()

        # Compute reward
        coverage = self._get_coverage()
        reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
        terminated = is_success = coverage > self.success_threshold

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["coverage"] = coverage
        
        # Add pressure information to info
        info['contact'] = self.get_contact_data() # this will be a dictionary with 'forces' and 'positions' relative to robot position of the applied force

        # print(info['contact_data'])
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()
        
        # Reset pressure data
        self.reset_pressure_data()

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            # Generate random state with workspace_size relative bounds
            rs = np.random.RandomState(seed=seed)
            margin = int(50)  # 10% margin from edges
            agent_min = margin
            agent_max = self.workspace_size - margin
            block_min = margin * 2  # 20% margin for blocks
            block_max = self.workspace_size - margin * 2  # 80% position for blocks

            state = np.array(
                [
                    rs.randint(agent_min, agent_max),
                    rs.randint(agent_min, agent_max),
                    rs.randint(block_min, block_max),
                    rs.randint(block_min, block_max),
                    rs.randn() * 2 * np.pi - np.pi,
                    rs.randint(block_min, block_max),
                    rs.randint(block_min, block_max),
                    rs.randn() * 2 * np.pi - np.pi,
                ],
                dtype=np.float64
            )
        self._set_state(state)

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = False

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((self.workspace_size, self.workspace_size))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)

        # # Draw goal pose
        # goal_body = self.get_goal_pose_body(self.goal_pose)
        # for shape in self.block.shapes:
        #     goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
        #     goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
        #     goal_points += [goal_points[0]]
        #     pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self._last_action is not None:
            action = np.array(self._last_action)
            coord = (action / self.workspace_size * [height, width]).astype(np.int32)
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        

        contact_data = self.get_contact_data()
        if len(contact_data['forces']) > 0:
            forces = contact_data['forces']
            positions = contact_data['positions'] + np.array(self.agent.position)  # Convert back to world coordinates
            

            min_force = np.min(forces) if np.min(forces) < np.max(forces) else 0
            max_force = np.max(forces) if np.max(forces) > min_force else 1
            normalized_forces = (forces - min_force) / (max_force - min_force) if max_force > min_force else np.ones_like(forces)
            normalized_forces = forces
            if self.render_contact:
                for i, (pos, norm_force) in enumerate(zip(positions, normalized_forces)):
                    # Convert position to image coordinates
                    pos_coord = (pos / self.workspace_size * [height, width]).astype(np.int32)
                    
                    # Color mapping: blue (low force) to red (high force) in BGR format for OpenCV
                    red = int(255 * norm_force)
                    blue = int(255 * (1 - norm_force))
                    green = 0
                    color = (blue, green, red)  # BGR format for OpenCV
                    # Draw contact point as a circle
                    radius = min(int(3 / 96 * render_size), int((3 * norm_force / 10) / 96 * render_size))
                    cv2.circle(img, pos_coord, radius, color, -1)
            
        # Add keypoints visualization
        if render_action and self.obs_type == "keypoints":
            # Draw block keypoints in red
            keypoints = self.get_keypoints(self._block_shapes)
            for point in keypoints:
                point_coord = (point / self.workspace_size * [height, width]).astype(np.int32)
                radius = int(1 / 96 * render_size)
                cv2.circle(img, point_coord, radius, (0, 0, 255), -1)  # Red circles
            
            # Draw goal keypoints in green
            keypoints_goal = self.get_keypoints(self._block_shapes_goal)
            for point in keypoints_goal:
                point_coord = (point / self.workspace_size * [height, width]).astype(np.int32)
                radius = int(1 / 96 * render_size)
                cv2.circle(img, point_coord, radius, (0, 255, 0), -1)  # Green circles
            
        return img

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        screen = self._draw()  # draw the environment on a screen

        if self.render_mode == "rgb_array":
            return self._get_img(screen, width=width, height=height, render_action=visualize)
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.workspace_size, self.workspace_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(
                screen, screen.get_rect()
            )  # copy our drawings from `screen` to the visible window
            
            # Visualize contact forces as colored dots
            contact_data = self.get_contact_data()
            if len(contact_data['forces']) > 0:
                forces = contact_data['forces']
                positions = contact_data['positions'] + np.array(self.agent.position)  # Convert back to world coordinates
                
                # Normalize forces for color mapping (0 to 1)
                if len(forces) > 0 and self.render_contact:
                    min_force = np.min(forces) if np.min(forces) < np.max(forces) else 0
                    max_force = np.max(forces) if np.max(forces) > min_force else 1
                    normalized_forces = (forces - min_force) / (max_force - min_force) if max_force > min_force else np.ones_like(forces)
                    # print("Max contact force:", max_force)
                    for i, (pos, norm_force) in enumerate(zip(positions, normalized_forces)):
                        # Color mapping: blue (low force) to red (high force)
                        # norm_force ranges from 0 to 1
                        blue = int(255 * norm_force)
                        red = int(255 * (1 - norm_force))
                        green = 0
                        color = (red, green, blue)
                        
                        # Draw contact point as a circle
                        radius = max(3, int(5 * norm_force + 2))  # Size also indicates force intensity
                        pygame.draw.circle(self.window, color, (int(pos[0]), int(pos[1])), radius)
            
            if self.obs_type == "keypoints":
                # add the key points that are stored in the observation 
                key_points = self.get_keypoints(self._block_shapes)
                for point in key_points:
                    pygame.draw.circle(self.window, pygame.Color("red"), point, 5)
                key_points_goal = self.get_keypoints(self._block_shapes_goal)
                for point in key_points_goal:
                    pygame.draw.circle(self.window, pygame.Color("green"), point, 5)
                if self._last_action is not None:
                    # Draw an X at the last action position
                    action_pos = self._last_action
                    # Create cross lines for X
                    line_length = 10
                    pygame.draw.line(
                        self.window, 
                        pygame.Color("red"),
                        (action_pos[0] - line_length, action_pos[1] - line_length),
                        (action_pos[0] + line_length, action_pos[1] + line_length),
                        5  # Increased line width from 3 to 5
                    )
                    pygame.draw.line(
                        self.window,
                        pygame.Color("red"),
                        (action_pos[0] - line_length, action_pos[1] + line_length),
                        (action_pos[0] + line_length, action_pos[1] - line_length),
                        5  # Increased line width from 3 to 5
                    )
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz)))
            pygame.display.update()

        else:
            raise ValueError(self.render_mode)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate([agent_position, block_position, [block_angle]], dtype=np.float64)

        if self.obs_type == "keypoints":
            return {
                "object_state" :np.array([self.block.position.x, self.block.position.y, self.block.angle]), #this angle is not blounded to 2*pi
                "goal_state": np.array([self.block_goal.position.x, self.block_goal.position.y, self.block_goal.angle]),
                "object_keypoints": self.get_keypoints(self._block_shapes).flatten(),
                "goal_keypoints": self.get_keypoints(self._block_shapes_goal).flatten(),
                "agent_pos": np.array(self.agent.position),
            }

        pixels = self._render()
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.array(self.agent.position),
            }

    @staticmethod
    def get_goal_pose_body(pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": np.array(list(self.block_goal.position) + [self.block_goal.angle]),
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _handle_collision(self, arbiter, space, data):
        """Enhanced collision handler that captures force and position data."""
        # Original contact counting
        self.n_contact_points += len(arbiter.contact_point_set.points)
        
        # Extract contact information
        contact_set = arbiter.contact_point_set
        
        for i, contact_point in enumerate(contact_set.points):
            # Get contact position in world coordinates
            contact_pos = contact_point.point_a
            
            # Calculate force magnitude from contact point data
            # Use distance as a proxy for force (closer contact = higher force)
            distance = contact_point.distance
            force_magnitude = max(0, 10.0 - abs(distance)) if distance is not None else 1.0
            
            # Store raw contact data
            self.contact_forces.append(force_magnitude)
            self.contact_positions.append([contact_pos.x, contact_pos.y])

    def _prevent_wall_penetration(self):
        """Prevent objects from penetrating walls by correcting positions."""
        wall_margin = 5
        wall_thickness = self.wall_thickness
        
        # Check block position and correct if it's outside bounds
        block_pos = self.block.position
        corrected_x = block_pos.x
        corrected_y = block_pos.y
        
        # Get approximate block size (T-shape extends about 60 units in each direction)
        block_size = 60
        
        # Check left wall
        if block_pos.x - block_size/2 < wall_margin + wall_thickness:
            corrected_x = wall_margin + wall_thickness + block_size/2
            self.block.velocity = Vec2d(max(0, self.block.velocity.x), self.block.velocity.y)
            
        # Check right wall  
        if block_pos.x + block_size/2 > self.workspace_size - wall_margin - wall_thickness:
            corrected_x = self.workspace_size - wall_margin - wall_thickness - block_size/2
            self.block.velocity = Vec2d(min(0, self.block.velocity.x), self.block.velocity.y)
            
        # Check bottom wall
        if block_pos.y - block_size/2 < wall_margin + wall_thickness:
            corrected_y = wall_margin + wall_thickness + block_size/2
            self.block.velocity = Vec2d(self.block.velocity.x, max(0, self.block.velocity.y))
            
        # Check top wall
        if block_pos.y + block_size/2 > self.workspace_size - wall_margin - wall_thickness:
            corrected_y = self.workspace_size - wall_margin - wall_thickness - block_size/2
            self.block.velocity = Vec2d(self.block.velocity.x, min(0, self.block.velocity.y))
            
        # Apply correction if needed
        if corrected_x != block_pos.x or corrected_y != block_pos.y:
            self.block.position = Vec2d(corrected_x, corrected_y)
        
        # Also check agent position
        agent_pos = self.agent.position
        agent_radius = 15
        agent_corrected_x = agent_pos.x
        agent_corrected_y = agent_pos.y
        
        # Check agent bounds
        if agent_pos.x - agent_radius < wall_margin + wall_thickness:
            agent_corrected_x = wall_margin + wall_thickness + agent_radius
        if agent_pos.x + agent_radius > self.workspace_size - wall_margin - wall_thickness:
            agent_corrected_x = self.workspace_size - wall_margin - wall_thickness - agent_radius
        if agent_pos.y - agent_radius < wall_margin + wall_thickness:
            agent_corrected_y = wall_margin + wall_thickness + agent_radius
        if agent_pos.y + agent_radius > self.workspace_size - wall_margin - wall_thickness:
            agent_corrected_y = self.workspace_size - wall_margin - wall_thickness - agent_radius
            
        # Apply agent correction if needed
        if agent_corrected_x != agent_pos.x or agent_corrected_y != agent_pos.y:
            self.agent.position = Vec2d(agent_corrected_x, agent_corrected_y)
            
    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        
        # Configure space for better collision detection and stability
        self.space.iterations = 30  # Increase collision resolution iterations
        self.space.collision_slop = 0.1  # Reduce collision slop for tighter collisions
        self.space.collision_bias = pow(1.0 - 0.9, 60.0)  # Improve collision bias
        
        self.teleop = False

        # Add walls - using workspace_size for positioning with thicker walls
        wall_margin = 5
        wall_end = self.workspace_size - wall_margin
        walls = [
            self.add_segment(self.space, (wall_margin, wall_end), (wall_margin, wall_margin), self.wall_thickness),
            self.add_segment(self.space, (wall_margin, wall_margin), (wall_end, wall_margin), self.wall_thickness),
            self.add_segment(self.space, (wall_end, wall_margin), (wall_end, wall_end), self.wall_thickness),
            self.add_segment(self.space, (wall_margin, wall_end), (wall_end, wall_end), self.wall_thickness),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone - positioned relative to workspace center
        center_x = self.workspace_size // 2
        center_y = self.workspace_size // 2
        self.agent = self.add_circle(self.space, (center_x, center_y + 100), 15)
        self.block, self._block_shapes = self.add_tee(self.space, (center_x, center_y), 0)
        # self.goal_pose = np.array([center_x, center_x, np.pi / 4])  # x, y, theta (in radians)
        self.block_goal, self._block_shapes_goal = self.add_tee(self.space, (center_x, center_y), 0, color="LightGreen", ghost=True)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling
        self.collision_handler = self.space.add_collision_handler(0, 0)
        self.collision_handler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _set_state(self, state):
        self.agent.position = list(state[:2])
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoretically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        self.block.angle = state[4]
        self.block.position = list(state[2:4])
        

        #set the goal pose
        self.block_goal.angle = state[7] 
        self.block_goal.position = list(state[5:7])
   
        self.goal_pose = np.array([state[5], state[6], state[7]])
        # Run physics to take effect
        self.space.step(self.dt)

    @staticmethod
    def add_segment(space, a, b, radius):
        # TODO(rcadene): rename add_segment to make_segment, since it is not added to the space
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
        shape.friction = 0.7  # Add friction to walls
        shape.elasticity = 0.0  # No bouncing off walls
        return shape

    @staticmethod
    def add_circle(space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        space.add(body, shape)
        return body

    @staticmethod
    def add_tee(space, position, angle, scale=30, color="LightSlateGray", mask=None, ghost=False):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        mass = 1
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        body.friction = 1
        if ghost:
            # Make it a ghost by setting special collision filters
            ghost_category = 0x2  # A unique category for ghost objects
            ghost_mask = 0x0      # Don't collide with anything
        else:
            ghost_category = 0x1  # Regular category
            ghost_mask = mask    # Regular mask

        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color) 
        shape2.color = pygame.Color(color)
        
        # Improve collision properties for stability
        if not ghost:
            shape1.friction = 0.7
            shape2.friction = 0.7
            shape1.elasticity = 0.0  # No bouncing
            shape2.elasticity = 0.0
        
        # Apply collision filters
        shape1.filter = pymunk.ShapeFilter(categories=ghost_category, mask=ghost_mask)
        shape2.filter = pymunk.ShapeFilter(categories=ghost_category, mask=ghost_mask)
        
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.angle = angle
        body.position = position
        space.add(body, shape1, shape2)
        return body, [shape1, shape2]
    @staticmethod
    def get_keypoints(block_shapes):
        """Get a (8, 2) numpy array with the T keypoints.

        The T is composed of two rectangles each with 4 keypoints.

        0───────────1
        │           │
        3───4───5───2
            │   │
            │   │
            │   │
            │   │
            7───6
        """
        keypoints = []
        for shape in block_shapes:
            for v in shape.get_vertices():
                v = v.rotated(shape.body.angle)
                v = v + shape.body.position
                keypoints.append(np.array(v))
            keypoints.append(keypoints[-4]*0.5 + keypoints[-3]*0.5)
            keypoints.append(keypoints[-3]*0.5 + keypoints[-2]*0.5)

        return np.row_stack(keypoints)