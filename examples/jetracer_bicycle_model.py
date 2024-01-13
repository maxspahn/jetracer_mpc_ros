import sys
import yaml
import pickle
import numpy as np

import gymnasium as gym
from urdfenvs.urdf_common.bicycle_model import BicycleModel
from jetracermpc.planner import MPCPlanner
from jetracermpc.model import JetRacerModel

class Simulation():

    _planner: MPCPlanner
    _state: np.ndarray
    _dt: float

    def __init__(self):
        self._planner = MPCPlanner(sys.argv[1])
        self._state = np.zeros(3)
        self._dt = self._planner.model.time_step
        self._planner.set_goal(np.array([3, 3]))
        self._planner.set_uniform_goal_weight(1)

    def run_dry(self):
        for i in range(100):
            action = self._planner.compute_action(self._state)
            xdot = np.array(self._planner.model.continuous_dynamics(self._state, action)).transpose()[0]
            self._state += self._dt * xdot
            print(self._state)

    def run(self):
        robots = [
            BicycleModel(
                urdf='prius.urdf',
                mode="vel",
                scaling=0.100,
                wheel_radius = 0.31265,
                wheel_distance = 0.494,
                spawn_offset = np.array([-0.435, 0.0, 0.05]),
                actuated_wheels=['front_right_wheel_joint', 'front_left_wheel_joint', 'rear_right_wheel_joint', 'rear_left_wheel_joint'],
                steering_links=['front_right_steer_joint', 'front_left_steer_joint'],
            )
        ]
        env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        )
        ob, *_ = env.reset()
        state = ob['robot_0']['joint_state']['position']
        for i in range(1000):
            action = self._planner.compute_action(state)
            xdot = np.array(self._planner.model.continuous_dynamics(state, action)).transpose()[0]
            predicted_state = state + 0.01 * xdot
            ob, *_ = env.step(action)
            state = ob['robot_0']['joint_state']['position']
            print(f"Observed state : {state}")
            print(f"Predicted state : {predicted_state}")
        env.close()



def main():
    sim = Simulation()
    sim.run()

if __name__ == "__main__":
    main()
