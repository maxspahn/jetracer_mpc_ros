from dataclasses import dataclass
import numpy as np
import yaml
import pickle
import forcespro
import casadi as ca
from shutil import move
from glob import glob

def diagSX(val, size):
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
    slack: bool
    interval: int
    number_obstacles: int
    model_length: float
    model_radius: float
    debug: bool = False



class JetRacerModel():
    def __init__(self, configuration: dict, init_param_map: bool = True):
        self._config = JetRacerConfiguration(**configuration)
        self._dt = self._config.time_step
        self._n_action = 2
        self._n_goal = 2
        self._n_state = 3
        self._n_slack = 1
        self._n_obst = self._config.number_obstacles
        self._m_obst = 3
        self._N = self._config.time_horizon
        self._npar = 0
        self.initParamMap()

    @property
    def n_all(self) -> int:
        return self._n_state + self._n_slack + self._n_action

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
        return self._paramMap

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

    def initParamMap(self):
        self._paramMap = {}
        self.addEntry2ParamMap("weight_action", self._n_action)
        self.addEntry2ParamMap("weight_goal", self._n_goal)
        self.addEntry2ParamMap("weight_slack", self._n_slack)
        self.addEntry2ParamMap("goal", self._n_goal)
        self.addEntry2ParamMap("obstacles", 4 * self._config.number_obstacles)
        self.addEntry2ParamMap('weight_obstacles', 1)

    def addEntry2ParamMap(self, name, n_par):
        self._paramMap[name] = list(range(self._npar, self._npar + n_par))
        self._npar += n_par

    def get_velocity(self, z):
        return  z[self._n: self._nx]

    def eval_objectiveCommon(self, z, p):
        state = z[0:3]
        front_offset = ca.vcat([ca.cos(state[2]), ca.sin(state[2])]) * self._config.model_length
        front_position = state[0:2] + front_offset
        weight_goal = p[self._paramMap["weight_goal"]]
        goal = p[self._paramMap["goal"]]
        W_goal = diagSX(weight_goal, self._n_goal)
        err = front_position - goal
        J_goal = ca.dot(err, ca.mtimes(W_goal, err))
        return J_goal
        Jobst = 0
        Js = 0
        obstDistances = 1/ca.vcat(self.eval_obstacleDistances(z, p) )
        wobst = ca.SX(np.ones(obstDistances.shape[0]) * p[self._paramMap['wobst']])
        Wobst = diagSX(wobst, obstDistances.shape[0])
        Jobst += ca.dot(obstDistances, ca.mtimes(Wobst, obstDistances))
        if self._ns > 0:
            s = z[self._nx]
            ws = p[self._paramMap["ws"]]
            Js += ws * s ** 2
        return Jx, Js, Jobst

    def eval_objectiveN(self, z, p):
        return self.eval_objectiveCommon(z, p)
        Jx, Jvel, Js, Jobst = self.eval_objectiveCommon(z, p)
        return Jx + Jvel + Js + Jobst

    def eval_objective(self, z, p):
        action = z[3:5]
        J_goal = self.eval_objectiveCommon(z, p)
        weight_action = p[self._paramMap["weight_action"]]
        W_action = diagSX(weight_action, self._n_action)
        J_action = ca.dot(action, ca.mtimes(W_action, action))
        return J_goal + J_action
        Jx, Jvel, Js, Jobst = self.eval_objectiveCommon(z, p)
        _, _, qddot, *_ = self.extractVariables(z)
        return Jx + Jvel + Js + Jobst + Ju

    def eval_inequalities(self, z, p):
        all_ineqs = self.eval_obstacleDistances(z, p)
        slack = z[self.n_all]
        for ineq in all_ineqs:
            ineq  += slack
        return all_ineqs

    def eval_obstacleDistances(self, z, p):
        ineqs = []
        state = z[0:3]
        slack = z[self.n_all]
        if "obst" in self._paramMap.keys():
            obsts = p[self._paramMap["obst"]]
            r_body = p[self._paramMap["r_body"]]
            for j, collision_link in enumerate(self._robot_config.collision_links):
                fk = self._fk.fk(
                    q,
                    self._robot_config.root_link,
                    collision_link,
                    positionOnly=True
                )[0:self._m]
                for i in range(self._config.number_obstacles):
                    obst = obsts[i * (self._m_obst + 1) : (i + 1) * (self._m_obst + 1)]
                    x = obst[0 : self._m_obst]
                    r = obst[self._m_obst]
                    dist = ca.norm_2(fk - x)
                    ineqs.append(dist - r - r_body)
        return ineqs


    def continuous_dynamics(self, state, action):
        """
        Assuming the simple bicycled model where state is composed of x, y, theta and action is forward velocity and steering angle.
        """
        xdot = ca.vcat([
            action[0]*ca.cos(state[2]),
            action[0]*ca.sin(state[2]),
            0.5 * action[0]*ca.tan(action[1])/self._config.model_length,
        ])
        return xdot

    def inititialize_forces_model(self):
        self._model = forcespro.nlp.SymbolicModel(self._N)
        self._model.continuous_dynamics = self.continuous_dynamics
        self._model.objective = self.eval_objective
        self._model.objectiveN = self.eval_objectiveN
        E = np.concatenate(
            [np.eye(self._n_state), np.zeros((self._n_state, self._n_action + self._n_slack))], axis=1
        )
        self._model.E = E
        self._model.lb = np.concatenate((
            self._config.lower_limits["state"],
            self._config.lower_limits["action"],
            [0],
        ))
        self._model.ub = np.concatenate((
            self._config.upper_limits["state"],
            self._config.upper_limits["action"],
            [np.inf],
        ))
        self._model.npar = self._npar
        self._model.nvar = self.n_all
        self._model.neq = self._n_state
        number_inequalities = 0
        number_inequalities += self._config.number_obstacles
        self._model.nh = number_inequalities
        self._model.hu = np.ones(number_inequalities) * np.inf
        self._model.hl = np.zeros(number_inequalities)
        self._model.ineq = self.eval_inequalities
        self._model.xinitidx = range(0, self._n_state)

    def set_codeoptions(self, **kwargs):
        solver_name = "jetracer_mpc_" + str(self._dt).replace('.','') + "_H" + str(self._N)
        if not self._config.slack:
            solver_name += "_noSlack"
        if solver_name in kwargs:
            solver_name = kwargs.get('solver_name')
        self._solver_name = solver_name
        self._codeoptions = forcespro.CodeOptions(solver_name)
        self._codeoptions.nlp.integrator.type = "ERK2"
        self._codeoptions.nlp.integrator.Ts = self._dt
        self._codeoptions.nlp.integrator.nodes = 5
        if self._config.debug:
            self._codeoptions.printlevel = 1
            self._codeoptions.optlevel = 0
        else:
            self._codeoptions.printlevel = 0
            self._codeoptions.optlevel = 3

    def save(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def generate_solver(self, location="./"):
        self.inititialize_forces_model()
        self.set_codeoptions()
        if self._config.debug:
            location += '_debug/'
        _ = self._model.generate_solver(self._codeoptions)
        self.save(self._solver_name + "/model")
        move(self._solver_name, location + self._solver_name)
        for file in glob(r'*.forces'):
            move(file, location)

