import sys
import time
from typing import List
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
import yaml
import pickle
import numpy as np

import gymnasium as gym
from urdfenvs.urdf_common.bicycle_model import BicycleModel
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from jetracermpc.planner import MPCPlanner
from jetracermpc.model import JetRacerModel

class Simulation():

    _planner: MPCPlanner
    _state: np.ndarray
    _dt: float
    _env: UrdfEnv
    _obstacles: List[List[float]]
    _radii: List[float]
    _goal: np.ndarray

    def __init__(self):
        self._planner = MPCPlanner(sys.argv[1])
        self._state = np.zeros(3)
        self._obstacles = []
        self._radii = []
        self.create_environment()
        self._planner.set_goal(self._goal)
        self._planner.set_obstacles(
            np.array(self._obstacles),
            np.array(self._radii),
        )
        self._planner.set_uniform_goal_weight(1)

    def create_environment(self):
        self._obstacles = [
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, -1.0, 0.0],
        ]
        self._radii = [
            0.3,
            0.5,
            0.1,
        ]
            
        self._dt = 0.05
        self._goal = np.array([float(sys.argv[2]), float(sys.argv[3])])
        robots = [
            BicycleModel(
                urdf='examples/racecar/racecar.urdf',
                mode="vel",
                scaling=1.000,
                wheel_radius = 0.31265,
                wheel_distance = 0.494,
                spawn_offset = np.array([-0.435, 0.0, 0.05]),
                actuated_wheels=[
                    'right_front_wheel_joint',
                    'left_front_wheel_joint',
                    'right_rear_wheel_joint',
                    'left_rear_wheel_joint'
                ],
                steering_links=[
                    'right_steering_hinge_joint',
                    'left_steering_hinge_joint',
                ],
            )
        ]
        self._env = UrdfEnv(
                robots,
                render=True,
                enforce_real_time = False,
                dt=self._dt,
                num_sub_steps=20,
                observation_checking=False
        )
        for obstacle_index, obstacle_position in enumerate(self._obstacles):
            radius = self._radii[obstacle_index]
            obstacle_data = {
                "type": "sphere",
                "movable": False,
                "geometry": {
                    "position": obstacle_position, 
                    "radius": radius
                },
            }
            obstacle = SphereObstacle(name=f"sphere_{obstacle_index}", content_dict=obstacle_data)
            self._env.add_obstacle(obstacle)
        self.init_path_visualization()

    def run_dry(self):
        for i in range(100):
            action = self._planner.compute_action(self._state)
            xdot = np.array(self._planner.model.continuous_dynamics(self._state, action)).transpose()[0]
            self._state += self._planner.model.time_step * xdot
            print(self._state)

    def init_path_visualization(self):
        goal_position = self._goal.tolist()
        goal_position.append(0)
        goal_dict = {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": 0,
            "child_link": 1,
            "desired_position": goal_position,
            "epsilon": 0.2,
            "type": "staticSubGoal",
        }
        self._env.add_goal(StaticSubGoal(name='goal', content_dict=goal_dict))
        for stage_id in range(self._planner.model.time_horizon):
            self._env.add_visualization(shape_type='sphere', size=[0.1])

    def update_path_visualization(self):
        predicted_states = self._planner.predicted_states()
        predicted_states[:, 2] *= 0
        self._env.update_visualizations(predicted_states)

    def continuous_dynamics(self, state, action):
        """
        Assuming the simple bicycled model where state is composed of x, y, theta and action is forward velocity and steering angle.
        """
        xdot = np.array([
            action[0]*np.cos(state[2]),
            action[0]*np.sin(state[2]),
            action[0]*np.tan(action[1])/0.7,
        ])
        return xdot
            

    def run(self):
        ob, *_ = self._env.reset()
        state = ob['robot_0']['joint_state']['position']
        for i in range(1000):
            t0 = time.perf_counter()
            action = self._planner.compute_action(state)
            self.update_path_visualization()
            predictions = self._planner.predicted_states()
            #xdot = self.continuous_dynamics(state, action)

            #predicted_state = state + 0.2 * self._dt * xdot
            ob, *_ = self._env.step(action)
            t1 = time.perf_counter()
            state = ob['robot_0']['joint_state']['position']
            steering = ob['robot_0']['joint_state']['steering']
            velocity = ob['robot_0']['joint_state']['forward_velocity']
            observed_action = np.array([velocity[0], steering[0]])
            #print(f"Steering : {steering}")
            #print(f"Observed state : {state[:3]}")
            #print(f"Predicted state : {predicted_state}")
            #print(f"Predicted state : {predictions[1][:3]}")
            #print(f"action : {action}")
            #print(f"observed_action: {observed_action}")
            compute_time = t1 - t0
            sleep_time = self._dt - compute_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._env.close()



def main():
    sim = Simulation()
    sim.run()

if __name__ == "__main__":
    main()
