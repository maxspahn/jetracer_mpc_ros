from typing import List
from jetracermpc.model import JetRacerModel
import pickle
import forcespro

import numpy as np


class EmptyObstacle:
    def position(self):
        return [-100, -100, -100]

    def radius(self):
        return -100

    def dim(self):
        return 3


class PlannerSettingIncomplete(Exception):
    pass


class MPCPlanner:
    _model: JetRacerModel
    _x0: np.ndarray
    _xinit: np.ndarray
    _params: np.ndarray

    def __init__(self, solver_folder: str):
        with open(solver_folder + "/model", "rb") as f:
            self._model = pickle.load(f)
        self._solver = forcespro.nlp.Solver.from_directory(solver_folder)
        self.reset()
        self._params = np.zeros(
            shape=(self._model.n_parameters * self._model.time_horizon),
            dtype=float,
        )
        self._action_counter = self._model.interval

    @property
    def model(self) -> JetRacerModel:
        return self._model

    def compute_action(self, x):
        """Computes control action, unless an interval is specified in
        which the action should be kept constrant."""
        if self._action_counter >= self._model.interval:
            self._action, info = self.solve(x)
            self._action_counter = 1
        else:
            self._action_counter += 1
        return self._action

    def reset(self):
        """Reset initial guess"""
        self._x0 = np.zeros((self._model.time_horizon, self._model.n_all))
        self._xinit = np.zeros(self._model.n_state)
        # self._x0[-1, -1] = 0.1

    def shift_horizon(self, output):
        """Apply receeding horizon to the commanded action."""
        for key in output.keys():
            if self._model.time_horizon < 10:
                stage = int(key[-1:])
            elif (
                self._model.time_horizon >= 10
                and self._model.time_horizon < 100
            ):
                stage = int(key[-2:])
            elif self._model.time_horizon >= 100:
                stage = int(key[-3:])
            if stage == 1:
                continue
            self._x0[stage - 2, 0 : len(output[key])] = output[key]

    def set_parameter_across_stages(self, parameter_index, value):
        """Set a parameter across all stages."""
        for stage_index in range(self.model.time_horizon):
            self._params[
                self.model.n_parameters * stage_index + parameter_index
            ] = value

    def set_parameters_across_stages(
        self, parameter_indices: List[int], values: np.ndarray
    ):
        """Set a vector parameter across all stages."""
        assert len(parameter_indices) == values.size
        for stage_index in range(self.model.time_horizon):
            indices = self.model.n_parameters * stage_index + np.array(
                parameter_indices
            )
            self._params[indices.tolist()] = values

    def set_uniform_goal_weight(self, value: float) -> None:
        for weight_index in range(self.model.n_goal):
            parameter_index = self.model.parameter_map["weight_goal"][
                weight_index
            ]
            self.set_parameter_across_stages(parameter_index, value)

    def set_action_weight(self, values: np.ndarray) -> None:
        parameter_indices = self.model.parameter_map["weight_action"]
        self.set_parameters_across_stages(parameter_indices, values)

    def set_uniform_obstacles_weight(self, value: float) -> None:
        for weight_index in range(1):
            parameter_index = self.model.parameter_map["weight_obstacles"][
                weight_index
            ]
            self.set_parameter_across_stages(parameter_index, value)

    def set_uniform_slack_weight(self, value: float) -> None:
        for weight_index in range(self.model.n_slack):
            parameter_index = self.model.parameter_map["weight_slack"][
                weight_index
            ]
            self.set_parameter_across_stages(parameter_index, value)

    def set_obstacles(
        self, obstacle_positions: np.ndarray, radii: np.ndarray
    ) -> None:
        """Set the obstacles across all stages."""
        assert obstacle_positions.shape[0] == self.model.number_obstacles
        assert obstacle_positions.shape[1] == self.model.size_obstacles
        assert radii.size == self.model.number_obstacles
        obstacle_parameters = np.concatenate(
            (obstacle_positions, radii[:, np.newaxis]), axis=1
        ).flatten()
        self.set_parameters_across_stages(
            self.model.parameter_map["obstacles"],
            obstacle_parameters,
        )

    def set_goal(self, goal_position: np.ndarray) -> None:
        assert goal_position.size == self.model.n_goal
        self.set_parameters_across_stages(
            self.model.parameter_map["goal"], goal_position
        )

    def solve(self, xinit):
        """Composes the optimization problem and calls the solver."""
        self._xinit = xinit
        action = np.zeros(self._model.n_action)
        problem = {}
        problem["xinit"] = self._xinit
        self._x0[0][0 : self._model.n_state] = self._xinit
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params
        output, exitflag, info = self._solver.solve(problem)
        if exitflag < 0:
            print(f"No solution found, exitflag : {exitflag}")
        if self.model.debug:
            print("----DEBUGGING----")
            npar = self.model.n_parameters
            for stage_id in range(self._model.time_horizon):
                p = self._params[stage_id * npar : stage_id * npar + npar]
                z = self._x0[stage_id]
                print(z[-1])
                print(self._model.common_objective(z, p))
                #print(self._model.inequalities(z, p))
            print("----DEBUGGING----")
        key1 = f"x{str(1).zfill(len(str(self._model.time_horizon)))}"
        action = output[key1][3:5]
        self.shift_horizon(output)
        return action, info

    def predicted_states(self) -> np.ndarray:
        """Returns the predicted states over the time horizon."""
        predicted_states = np.zeros(
            (self.model.time_horizon, self.model.n_state)
        )
        for stage_id in range(self.model.time_horizon):
            predicted_states[stage_id, :] = self._x0[stage_id][
                0: self.model.n_state
            ]
        return predicted_states
