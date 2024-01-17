from typing import Optional
from dataclasses import dataclass
import numpy as np
import pickle
import forcespro
import casadi as ca
from shutil import move
from glob import glob


def diagSX(val, size):
    """Creates casadi SX matrix based an a vector of values.
    The values are then on the diagonal of the matrix.
    """
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a


@dataclass
class JetRacerConfiguration:
    time_horizon: int
    time_step: float
    lower_limits: dict
    upper_limits: dict
    interval: int
    number_obstacles: int
    model_length: float
    model_radius: float
    debug: bool = False


class JetRacerModel:
    def __init__(self, configuration: dict):
        self._config = JetRacerConfiguration(**configuration)
        self._n_action = 2
        self._n_goal = 2
        self._n_state = 3
        self._n_slack = 1
        self._size_obstacles = 3
        self._npar = 0
        self.init_parameter_map()

    @property
    def n_all(self) -> int:
        return self._n_state + self._n_slack + self._n_action

    @property
    def n_slack(self) -> int:
        return self._n_slack

    @property
    def number_obstacles(self) -> int:
        return self._config.number_obstacles

    @property
    def size_obstacles(self) -> int:
        return self._size_obstacles

    @property
    def n_state(self) -> int:
        return self._n_state

    @property
    def debug(self) -> bool:
        return self._config.debug

    @property
    def n_action(self) -> int:
        return self._n_action

    @property
    def n_goal(self) -> int:
        return self._n_goal

    @property
    def parameter_map(self) -> dict:
        return self._parameter_map

    @property
    def interval(self) -> int:
        return self._config.interval

    @property
    def n_parameters(self) -> int:
        return self._npar

    @property
    def time_horizon(self) -> int:
        return self._config.time_horizon

    @property
    def time_step(self) -> float:
        return self._config.time_step

    def init_parameter_map(self):
        """Intializes a parameter map that hold information about the
        indices of individual parameters based on the parameter name.

        ForcesPro relies on a single parameter vector to pass all parameters
        to the solver. As this can be hard to keep track of, this method creates
        a dictionary to store the information of the indices of the parameters
        by name, such that interaction is simplified.
        """
        self._parameter_map = {}
        self.add_parameter("weight_action", self._n_action)
        self.add_parameter("weight_goal", self._n_goal)
        self.add_parameter("weight_slack", self._n_slack)
        self.add_parameter("goal", self._n_goal)
        self.add_parameter("obstacles", 4 * self._config.number_obstacles)
        self.add_parameter("weight_obstacles", 1)

    def add_parameter(
        self, parameter_name: str, parameter_dimension: int
    ) -> None:
        """Adds one parameter to the parameter_map."""
        self._parameter_map[parameter_name] = list(
            range(self._npar, self._npar + parameter_dimension)
        )
        self._npar += parameter_dimension

    def common_objective(self, z, p):
        """Evaluates those objective terms that are shared accross all stages."""
        state = z[0:3]
        front_offset = (
            ca.vcat([ca.cos(state[2]), ca.sin(state[2])])
            * self._config.model_length
        )
        front_position = state[0:2] + front_offset
        weight_goal = p[self._parameter_map["weight_goal"]]
        goal = p[self._parameter_map["goal"]]
        W_goal = diagSX(weight_goal, self._n_goal)
        err = front_position - goal
        J_goal = ca.dot(err, ca.mtimes(W_goal, err))
        obstacle_distances = 1 / ca.vcat(self.obstacle_distances(z, p))
        weight_obstacles = ca.SX(
            np.ones(obstacle_distances.shape[0]) * p[self._parameter_map["weight_obstacles"]]
        )
        W_obstacles = diagSX(weight_obstacles, obstacle_distances.shape[0])
        J_obstacles = ca.dot(obstacle_distances, ca.mtimes(W_obstacles, obstacle_distances))
        slack = z[self.n_all-1]
        weight_slack = p[self._parameter_map["weight_slack"]]
        J_slack = weight_slack * slack**2
        return J_goal + J_obstacles + J_slack

    def obective_last_stage(self, z, p):
        """Evaluates the objective on the last stage."""
        return self.common_objective(z, p)

    def objective(self, z, p):
        """Evaluates the objective for all stages except the last."""
        action = z[0:2]
        J_common = self.common_objective(z, p)
        weight_action = p[self._parameter_map["weight_action"]]
        W_action = diagSX(weight_action, self._n_action)
        J_action = ca.dot(action, ca.mtimes(W_action, action))
        return J_common + J_action

    def inequalities(self, z, p) -> list:
        """Evaluates the inequalities."""
        all_ineqs = self.obstacle_distances(z, p)
        slack = z[self.n_all - 1]
        for ineq in all_ineqs:
            ineq += slack
        return all_ineqs

    def obstacle_distances(self, z, p) -> list:
        """Evalutase the distance to obstacles."""
        ineqs = []
        state = z[0:3]
        obstacle_parameters = p[self._parameter_map["obstacles"]]
        parameters_per_obstacle = self.size_obstacles + 1
        for obstacle_index in range(self._config.number_obstacles):
            start_index = obstacle_index * parameters_per_obstacle
            end_index = start_index + parameters_per_obstacle
            obstacle_i_parameters = obstacle_parameters[start_index:end_index]
            position = obstacle_i_parameters[0 : self.size_obstacles]
            radius = obstacle_i_parameters[self.size_obstacles]
            distance = ca.norm_2(state[0:2] - position[0:2])
            ineqs.append(distance - radius - self._config.model_length)
        return ineqs

    def continuous_dynamics(self, state, action, parameters):
        """
        Assuming the simple bicycled model where state is composed of x, y,
        theta and action is forward velocity and steering angle.
        """
        xdot = ca.vcat(
            [
                action[0] * ca.cos(state[2]),
                action[0] * ca.sin(state[2]),
                0.5
                * action[0]
                * ca.tan(action[1])
                / self._config.model_length,
            ]
        )
        return xdot

    def inititialize_forces_model(self):
        self._model = forcespro.nlp.SymbolicModel(self.time_horizon)
        self._model.continuous_dynamics = self.continuous_dynamics
        self._model.objective = self.objective
        self._model.objectiveN = self.obective_last_stage
        E = np.concatenate(
            [
                np.eye(self._n_state),
                np.zeros((self._n_state, self._n_action + self._n_slack)),
            ],
            axis=1,
        )
        self._model.E = E
        self._model.lb = np.concatenate(
            (
                self._config.lower_limits["state"],
                self._config.lower_limits["action"],
                [0],
            )
        )
        self._model.ub = np.concatenate(
            (
                self._config.upper_limits["state"],
                self._config.upper_limits["action"],
                [np.inf],
            )
        )
        self._model.npar = self._npar
        self._model.nvar = self.n_all
        self._model.neq = self._n_state
        number_inequalities = 0
        number_inequalities += self._config.number_obstacles
        self._model.nh = number_inequalities
        self._model.hu = np.ones(number_inequalities) * np.inf
        self._model.hl = np.zeros(number_inequalities)
        self._model.ineq = self.inequalities
        self._model.xinitidx = [0, 1, 2]

    def set_codeoptions(self, solver_name: str):
        self._codeoptions = forcespro.CodeOptions(solver_name)
        self._codeoptions.nlp.integrator.type = "ERK2"
        self._codeoptions.nlp.integrator.Ts = self.time_step
        self._codeoptions.nlp.integrator.nodes = 5
        if self._config.debug:
            self._codeoptions.printlevel = 1
            self._codeoptions.optlevel = 0
        else:
            self._codeoptions.printlevel = 0
            self._codeoptions.optlevel = 3

    def save(self, file_name: str):
        """Saves model as pickle."""
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    def generate_solver(
        self, solver_name: Optional[str] = None, location="./"
    ):
        """Generates the solver and stores the model in the same folder."""
        if not solver_name:
            solver_name = (
                "jetracer_mpc_"
                + str(self.time_step).replace(".", "")
                + "_H"
                + str(self.time_horizon)
            )
        self.inititialize_forces_model()
        self.set_codeoptions(solver_name)
        if self._config.debug:
            location += "_debug/"
        _ = self._model.generate_solver(self._codeoptions)
        self.save(solver_name + "/model")
        move(solver_name, location + solver_name)
        for file in glob(r"*.forces"):
            move(file, location)
