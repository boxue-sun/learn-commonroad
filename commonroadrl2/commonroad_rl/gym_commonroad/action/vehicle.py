""" Module for managing the vehicle in the CommonRoad Gym environment
"""
import copy

import numpy as np
import commonroad_dc.pycrcc as pycrcc
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from aenum import extend_enum
from scipy.optimize import Bounds
from commonroad.scenario.trajectory import State, CustomState
from commonroad_rl.gym_commonroad.utils.scenario import make_valid_orientation
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, FrictionCircleException
from vehiclemodels.vehicle_parameters import VehicleParameters
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
from scipy.integrate import odeint

N_INTEGRATION_STEPS = 100

extend_enum(VehicleModel, 'YawRate', len(VehicleModel))
extend_enum(VehicleModel, 'QP', len(VehicleModel))
extend_enum(VehicleModel, 'PMNonlinear', len(VehicleModel))


# Using VehicleParameterMapping from feasibility checker causes bugs
def to_vehicle_parameter(vehicle_type: VehicleType):
    if vehicle_type == VehicleType.FORD_ESCORT:
        return parameters_vehicle1()
    elif vehicle_type == VehicleType.BMW_320i:
        return parameters_vehicle2()
    elif vehicle_type == VehicleType.VW_VANAGON:
        return parameters_vehicle3()
    else:
        raise TypeError(f"Vehicle type {vehicle_type} not supported!")


def assert_vehicle_model(vehicle_model: VehicleModel):
    if vehicle_model == VehicleModel.MB:
        raise NotImplementedError(f"Vehicle model {vehicle_model} is not implemented yet!")
    else:
        return vehicle_model


class Vehicle(ABC):
    """
    Description:
        Abstract base class of all vehicles
    """

    def __init__(self, params_dict: dict) -> None:
        """ Initialize empty object """
        vehicle_type = VehicleType(params_dict["vehicle_type"])
        vehicle_model = VehicleModel(params_dict["vehicle_model"])

        self.vehicle_type = vehicle_type
        self.vehicle_model = assert_vehicle_model(vehicle_model)
        self.parameters = to_vehicle_parameter(vehicle_type)
        self.name = None
        self.dt = None
        self._collision_object = None
        self.initial_state = None
        self.state_list = None

    @property
    def state(self) -> State:
        """
        Get the current state of the vehicle

        :return: The current state of the vehicle
        """
        return self.state_list[-1]

    @property
    def previous_state(self) -> State:
        """
        Get the previous state of the vehicle

        :return: The previous state of the vehicle
        """
        if len(self.state_list) > 1:
            return self.state_list[-2]
        else:
            return self.initial_state

    @state.setter
    def state(self, state: State):
        """ Set the current state of the vehicle is not supported """
        raise ValueError("To set the state of the vehicle directly is prohibited!")

    @property
    def collision_object(self) -> pycrcc.RectOBB:
        """
        Get the collision object of the vehicle

        :return: The collision object of the vehicle
        """
        return self._collision_object

    @collision_object.setter
    def collision_object(self, collision_object: pycrcc.RectOBB):
        """ Set the collision_object of the vehicle is not supported """
        raise ValueError("To set the collision_object of the vehicle directly is prohibited!")

    @property
    def current_time_step(self):
        return self.state.time_step

    @current_time_step.setter
    def current_time_step(self, current_time_step):
        raise ValueError("To set the current time step of the vehicle directly is prohibited!")

    def create_obb_collision_object(self, state: State):
        return pycrcc.RectOBB(self.parameters.l / 2,
                              self.parameters.w / 2,
                              state.orientation,
                              state.position[0],
                              state.position[1])

    def update_collision_object(self, create_convex_hull=True):
        """ Updates the collision_object of the vehicle """
        if create_convex_hull:
            self._collision_object = pycrcc.TimeVariantCollisionObject(self.previous_state.time_step)
            self._collision_object.append_obstacle(self.create_obb_collision_object(self.previous_state))
            self._collision_object.append_obstacle(self.create_obb_collision_object(self.state))
            self._collision_object, err = trajectory_queries.trajectory_preprocess_obb_sum(self._collision_object)
            if not err:
                return
            raise Exception("trajectory preprocessing error")
        else:
            self._collision_object = pycrcc.TimeVariantCollisionObject(self.state.time_step)
            self._collision_object.append_obstacle(self.create_obb_collision_object(self.state))

    @abstractmethod
    def set_current_state(self, new_state: State):
        """
        Update state list
        """
        raise NotImplementedError

    def reset(self, initial_state: State, dt: float) -> None:
        """
        Reset vehicle parameters.

        :param initial_state: The initial state of the vehicle
        :param dt: Simulation dt of the scenario
        :return: None
        """
        self.dt = dt
        if self.vehicle_model == VehicleModel.PM:
            orientation = initial_state.orientation if hasattr(initial_state, "orientation") else 0.0
            self.initial_state = CustomState(**{"position": initial_state.position,
                                          "orientation": make_valid_orientation(orientation),
                                          "time_step": initial_state.time_step,
                                          "velocity": initial_state.velocity * np.cos(orientation),
                                          "velocity_y": initial_state.velocity * np.sin(orientation),
                                          "acceleration": initial_state.acceleration * np.cos(orientation)
                                          if hasattr(initial_state, "acceleration") else 0.0,
                                          "acceleration_y": initial_state.acceleration * np.sin(orientation)
                                          if hasattr(initial_state, "acceleration") else 0.0})
        elif self.vehicle_model == VehicleModel.QP:
            self.initial_state = CustomState(**{"position": initial_state.position,
                                          "velocity": initial_state.velocity,
                                          "acceleration": initial_state.acceleration
                                          if hasattr(initial_state, "acceleration")
                                          else 0.0,
                                          "jerk": initial_state.jerk
                                          if hasattr(initial_state, "jerk") else 0.0,
                                          "orientation": make_valid_orientation(initial_state.orientation)
                                          if hasattr(initial_state, "orientation")
                                          else 0.0,
                                          "slip_angle": 0.0,
                                          "yaw_rate": 0.0,
                                          "time_step": initial_state.time_step,
                                          })
        else:
            self.initial_state = CustomState(**{"position": initial_state.position,
                                          "steering_angle": initial_state.steering_angle
                                          if hasattr(initial_state, "steering_angle")
                                          else 0.0,
                                          "orientation": make_valid_orientation(initial_state.orientation)
                                          if hasattr(initial_state, "orientation")
                                          else 0.0,
                                          "yaw_rate": initial_state.yaw_rate
                                          if hasattr(initial_state, "yaw_rate")
                                          else 0.0,
                                          "time_step": initial_state.time_step,
                                          "velocity": initial_state.velocity,
                                          "acceleration": initial_state.acceleration
                                          if hasattr(initial_state, "acceleration")
                                          else 0.0})
        self.state_list: List[State] = [self.initial_state]
        self.update_collision_object(create_convex_hull=self._continuous_collision_checking)

    def rescale_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range

        :param normalized_action: action from the CommonroadEnv.
        :return: rescaled action
        """
        pass


class ContinuousVehicle(Vehicle):
    """
    Description:
        Class for vehicle when trained in continuous action space
    """

    def __init__(self, params_dict: dict, continuous_collision_checking=True):
        """ Initialize empty object """
        super().__init__(params_dict)
        self.violate_friction = False
        self.jerk_bounds = np.array([-10000, 10000])
        self._continuous_collision_checking = continuous_collision_checking

        try:
            self.vehicle_dynamic = VehicleDynamics.from_model(self.vehicle_model, self.vehicle_type)
        except:
            if self.vehicle_model == VehicleModel.YawRate:
                # customize YawRate VehicleModel
                self.vehicle_dynamic = YawRateDynamics(self.vehicle_type)
                self.parameters = self.vehicle_dynamic.parameters
            elif self.vehicle_model == VehicleModel.QP:
                self.vehicle_dynamic = QPDynamics(self.vehicle_type)
            elif self.vehicle_model == VehicleModel.PMNonlinear:
                self.vehicle_dynamic = PointMassNonlinearDynamics(self.vehicle_type)
            else:
                raise ValueError(f"Unknown vehicle model: {self.vehicle_model}")

    def set_current_state(self, new_state: State):
        """
        Update state list

        :param new_state: new state
        :return: None
        """
        self.state_list.append(new_state)
        self.update_collision_object(create_convex_hull=self._continuous_collision_checking)

    def propagate_one_time_step(self, current_state: State, action: np.ndarray, action_base: str) -> State:
        """Generate the next state from a given state for the given action.

        :param current_state: current state of vehicle to propagate from
        :param action: control inputs of vehicle (real input)
        :param action_base: aspect on which the action should be based ("jerk", "acceleration")
        :return: propagated state
        """
        if action_base == "acceleration":
            u_input = action
        elif action_base == "jerk":
            u_input = self._jerk_to_acc(action)
        else:
            raise ValueError(f"Unknown action base: {action_base}")

        if self.vehicle_model == VehicleModel.PM:
            # using vehicle_dynamics.state_to_array(current_state) causes error since state has orientation and velocity
            x_current = np.array(
                [
                    current_state.position[0],
                    current_state.position[1],
                    current_state.velocity,
                    current_state.velocity_y,
                ]
            )

            # if maximum absolute acceleration is exceeded, rescale the acceleration
            absolute_acc = u_input[0] ** 2 + u_input[1] ** 2
            if absolute_acc > self.parameters.longitudinal.a_max ** 2:
                rescale_factor = (self.parameters.longitudinal.a_max - 1e-6) / np.sqrt(absolute_acc)
                # rescale the acceleration to satisfy friction circle constraint
                u_input[0] *= rescale_factor
                u_input[1] *= rescale_factor
        else:
            x_current, _ = self.vehicle_dynamic.state_to_array(current_state)

        try:
            x_current_old = copy.deepcopy(x_current)
            x_current = self.vehicle_dynamic.forward_simulation(x_current, u_input, self.dt, throw=True)
            self.violate_friction = False
        except FrictionCircleException:
            self.violate_friction = True
            for _ in range(N_INTEGRATION_STEPS):
                # simulate state transition - t parameter is set to vehicle.dt but irrelevant for the current vehicle models
                # TODO：x_dot of KS model considers the action constraints, which YR and PM model have not included yet
                x_dot = np.array(self.vehicle_dynamic.dynamics(self.dt, x_current, u_input))
                # update state
                x_current = x_current + x_dot * (self.dt / N_INTEGRATION_STEPS)

        # feed in required slots
        if self.vehicle_model == VehicleModel.PM:
            kwarg = {
                "position": np.array([x_current[0], x_current[1]]),
                "velocity": x_current[2],
                "velocity_y": x_current[3],
                "acceleration": u_input[0],
                "acceleration_y": u_input[1],
                "orientation": make_valid_orientation(np.arctan2(x_current[3], x_current[2])),
                "time_step": current_state.time_step + 1,
            }
        elif self.vehicle_model == VehicleModel.KS:
            state = self.vehicle_dynamic.array_to_state(x_current, time_step=current_state.time_step + 1)
            state.orientation = make_valid_orientation(state.orientation)
            state.acceleration = u_input[1]
            state.yaw_rate = (x_current[4] - x_current_old[4]) / self.dt

            return state
        elif self.vehicle_model == VehicleModel.YawRate:
            kwarg = {
                "position": np.array([x_current[0], x_current[1]]),
                "steering_angle": x_current[2],
                "velocity": x_current[3],
                "orientation": make_valid_orientation(x_current[4]),
                "acceleration": u_input[1],
                "yaw_rate": u_input[0],
                "time_step": current_state.time_step + 1,
            }
        elif self.vehicle_model == VehicleModel.QP:
            state = self.vehicle_dynamic.array_to_state(x_current, time_step=current_state.time_step + 1)
            # TODO velocity_z and roll_rate are used to jounce and kappa_dot_dot
            state.velocity_z = u_input[0]
            state.roll_rate = u_input[1]

            return state
        elif self.vehicle_model == VehicleModel.PMNonlinear:
            kwarg = {
                "position": np.array([x_current[0], x_current[1]]),
                "velocity": x_current[2],
                "acceleration": u_input[0],
                "orientation": make_valid_orientation(x_current[3]),
                "yaw_rate": u_input[1],
                "time_step": current_state.time_step + 1,
            }

        return CustomState(**kwarg)

    def get_new_state(self, action: np.ndarray, action_base: str) -> State:
        """Generate the next state from current state for the given action.

        :params action: rescaled action
        :params action_base: aspect on which the action should be based ("jerk", "acceleration")
        :return: next state of vehicle"""

        current_state = self.state

        return self.propagate_one_time_step(current_state, action, action_base)

    def _jerk_to_acc(self, action: np.ndarray) -> np.ndarray:
        """
        computes the acceleration based input on jerk based actions
        :param action: action based on jerk
        :return: input based on acceleration
        """
        if self.vehicle_model == VehicleModel.PM:
            # action[jerk_x, jerk_y]
            action = np.array([np.clip(action[0], self.jerk_bounds[0], self.jerk_bounds[1]),
                               np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input = np.array([self.state.acceleration + action[0] * self.dt,
                                self.state.acceleration_y + action[1] * self.dt])

        elif self.vehicle_model == VehicleModel.KS:
            # action[steering angel speed, jerk]
            action = np.array([action[0], np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input = np.array([action[0], self.state.acceleration + action[1] * self.dt])

        elif self.vehicle_model == VehicleModel.YawRate:
            # action[yaw, jerk]
            action = np.array([action[0], np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input = np.array([action[0], self.state.acceleration + action[1] * self.dt])
        else:
            raise ValueError(f"Unknown vehicle model: {self.vehicle_model}")

        return u_input


class YawParameters():
    def __init__(self):
        # constraints regarding yaw
        self.v_min = []  # minimum yaw velocity [rad/s]
        self.v_max = []  # maximum yaw velocity [rad/s]


def extend_vehicle_params(p: VehicleParameters) -> VehicleParameters:
    p.yaw = YawParameters()
    p.yaw.v_min = -2.  # minimum yaw velocity [rad/s]
    p.yaw.v_max = 2.  # maximum yaw velocity [rad/s]
    return p


class YawRateDynamics(VehicleDynamics):
    """
    Description:
        Class for the calculation of vehicle dynamics of YawRate vehicle model
    """

    def __init__(self, vehicle_type: VehicleType):
        super(YawRateDynamics, self).__init__(VehicleModel.YawRate, vehicle_type)
        self.parameters = extend_vehicle_params(self.parameters)
        self.wheelbase = self.parameters.a + self.parameters.b

        self.velocity = None

    def dynamics(self, t, x, u) -> List[float]:
        """
        Yaw Rate model dynamics function.

        :param x: state values, [position x, position y, steering angle, longitudinal velocity, orientation(yaw angle)]
        :param u: input values, [yaw rate, longitudinal acceleration]

        :return: system dynamics
        """
        velocity_x = x[3] * np.cos(x[4])
        velocity_y = x[3] * np.sin(x[4])
        self.velocity = x[3]

        # steering angle velocity depends on longitudinal velocity and yaw rate (as well as vehicle parameters)
        steering_ang_velocity = -u[0] * self.wheelbase / (x[3] ** 2 + u[0] * self.wheelbase ** 2)

        return [velocity_x, velocity_y, steering_ang_velocity, u[1], u[0]]

    @property
    def input_bounds(self) -> Bounds:
        """
        Overrides the bounds method of Vehicle Model in order to return bounds for the Yaw Rate Model inputs.

        Bounds are
            - -max longitudinal acc <= acceleration <= max longitudinal acc
            - mini yaw velocity <= yaw_rate <= max yaw velocity

        :return: Bounds
        """
        return Bounds([self.parameters.yaw.v_min - 1e-4, -self.parameters.longitudinal.a_max],
                      [self.parameters.yaw.v_max + 1e-4, self.parameters.longitudinal.a_max])

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0],
            state.position[1],
            getattr(state, 'steering_angle', steering_angle_default),  # not defined in initial state
            state.velocity,
            state.orientation,
        ]
        return np.array(values), state.time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
        }
        state = CustomState(**values, time_step=time_step)
        return state

    def _input_to_array(self, input: State) -> Tuple[np.array, int]:
        """
        Actual conversion of input to array happens here. Vehicles can override this method to implement their own converter.
        """
        values = [
            input.yaw_rate,
            input.acceleration,
        ]
        return np.array(values), input.time_step

    def _array_to_input(self, u: np.array, time_step: int) -> State:
        """
        Actual conversion of input array to input happens here. Vehicles can override this method to implement their
        own converter.
        """
        values = {
            'yaw_rate': u[0],
            'acceleration': u[1],
        }
        return CustomState(**values, time_step=time_step)


class QPDynamics(VehicleDynamics):
    """
    Description:
        Class for the calculation of vehicle dynamics of YawRate vehicle model
    """

    def __init__(self, vehicle_type: VehicleType):
        super(QPDynamics, self).__init__(VehicleModel.QP, vehicle_type)
        self.theta_ref = 0.

    @property
    def input_bounds(self) -> Bounds:
        """
        Overrides the bounds method of Vehicle Model in order to return bounds for the Yaw Rate Model inputs.

        Bounds are
            - max jerk_dot <= jerk_dot <= max jerk_dot
            - min kappa_dot_dot <= kappa_dot_dot <= max kappa_dot_dot

        :return: Bounds
        """
        return Bounds(
            np.array([-1000., -20.]),
            np.array([1000., 20.])
        )

    @staticmethod
    def jerk_dot_constraints(jerk_dot, acceleration, p):
        if (jerk_dot < 0. and acceleration <= -p.a_max) or \
                (jerk_dot > 0. and acceleration >= p.a_max):
            jerk_dot = 0.
        # TODO: integrate jerk dot constrain in vehicle parameters
        return jerk_dot

    @staticmethod
    def kappa_dot_dot_constraints(kappa_dot_dot, kappa_dot, kappa_dot_min, kappa_dot_max):
        if (kappa_dot < kappa_dot_min and kappa_dot_dot < 0.) or (kappa_dot > kappa_dot_max and kappa_dot_dot > 0.):
            kappa_dot_dot = 0.
        # TODO: integrate jerk dot constrain in vehicle parameters
        return kappa_dot_dot

    @staticmethod
    def vehicle_dynamics_linear(x_init, u_init, kappa_dot_max, p, theta_ref):
        x = np.array([x_init[:4], x_init[4:]])
        u = []
        u.append(QPDynamics.jerk_dot_constraints(u_init[0], x[0, 2], p.longitudinal))
        u.append(QPDynamics.kappa_dot_dot_constraints(u_init[1], x[1, 3], -kappa_dot_max, kappa_dot_max))

        x_long = x[0, :]
        x_lat = x[1, :]

        # longitudinal dynamics
        f_long = [
            x_long[1],
            x_long[2],
            x_long[3],
            u[0]
        ]

        # lateral dynamics
        v = x_long[1]  # TODO: QP planner uses the propagated velocity

        f_lat = [
            v * x_lat[1] - v * theta_ref,
            v * x_lat[2],
            x_lat[3],
            u[1]
        ]

        # f = [f_long, f_lat]
        f = f_long + f_lat

        return f

    def discrete_dynamics(self, x, u_init, dt, kappa_dot_max):
        x = np.array([x[:4], x[4:]])
        u = []
        u.append(QPDynamics.jerk_dot_constraints(u_init[0], x[0, 2], self.parameters.longitudinal))
        u.append(QPDynamics.kappa_dot_dot_constraints(u_init[1], x[1, 3], -kappa_dot_max, kappa_dot_max))

        x_long = x[0, :]
        x_lat = x[1, :]

        A_long = np.array(
            [[1, dt, (dt ** 2.) / 2., (dt ** 3.) / 6.],
             [0, 1., dt, (dt ** 2.) / 2.],
             [0, 0, 1., dt],
             [0, 0, 0, 1]]
        )
        B_long = np.array([[(dt ** 4.) / 24.], [(dt ** 3.) / 6.], [(dt ** 2.) / 2.], [dt]])

        v = x_long[1]
        A_lat = np.array(
            [
                [1, dt * v, (dt ** 2) * 0.5 * (v ** 2), (dt ** 3) / 6 * (v ** 2)],
                [0, 1, dt * v, (dt ** 2) * 0.5 * v],
                [0, 0, 1, dt],
                [0, 0, 0, 1]
            ])

        B_lat = np.array([[(dt ** 4) / 24 * (v ** 2)],
                          [(dt ** 3) / 6 * v],
                          [(dt ** 2) * 0.5],
                          [dt]])

        x_long = np.dot(A_long, x_long) + np.squeeze(B_long * u[0])
        x_lat = np.dot(A_lat, x_lat) + np.squeeze(B_lat * u[1])

        return np.concatenate([x_long, x_lat])

    def dynamics(self, t, x, u_init, dt) -> List[float]:
        # wb = self.parameters.a + self.parameters.b
        # kappa_max = np.tan(self.parameters.steering.max) / wb
        kappa_max = 0.2
        kappa_dot_max = 2 * kappa_max / dt
        self.theta_ref = 0.  # TODO: integrate reference path
        return QPDynamics.vehicle_dynamics_linear(x, u_init, kappa_dot_max, self.parameters, self.theta_ref)

        # # discrete integration
        # return self.discrete_dynamics(x, u_init, dt, kappa_dot_max)

    def forward_simulation(self, x: np.array, u_init: np.array, dt: float, throw: bool = True) -> np.array:
        """
        Simulates the next state using the given state and input values as numpy arrays.

        :param x: state values.
        :param u: input values
        :param dt: scenario delta time.
        :param throw: if set to false, will return None as next state instead of throwing exception (default=True)
        :return: simulated next state values, raises VehicleDynamicsException if invalid input.
        """

        x0, x1 = odeint(self.dynamics, x, [0.0, dt], args=(u_init, dt, ), tfirst=True)
        # x1 = self.dynamics(None, x, u_init, dt)

        return x1

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        # (s, v, a, j)  (d, θ, κ,˙κ)
        theta_ref = 0.
        # TODO: convert Cartesian position to CCOSY position
        values = np.array(
            [state.position[0] - self.parameters.b * np.cos(state.orientation),
             state.velocity * np.cos(state.orientation - self.theta_ref), state.acceleration, state.jerk] +
            [state.position[1] - self.parameters.b * np.sin(state.orientation), state.orientation, state.slip_angle,
             state.yaw_rate]
        )  # TODO slip_angle and yaw_rate are used to store kappa and kappa_dot

        return values, state.time_step

    def _array_to_state(self, x_init: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        # TODO: convert CCOSY position to Cartesian position
        # (s, v, a, j)  (d, θ, κ,˙κ)
        x = np.array([x_init[:4], x_init[4:]])
        values = {
            'position': np.array([x[0, 0] + self.parameters.b * np.cos(x[1, 1]),
                                  x[1, 0] + self.parameters.b * np.sin(x[1, 1])]),
            'velocity': x[0, 1] / np.cos(x[1, 1] - self.theta_ref),
            'acceleration': x[0, 2],
            'jerk': x[0, 3],
            'orientation': make_valid_orientation(x[1, 1]),
            'slip_angle': x[1, 2],
            'yaw_rate': x[1, 3],
        }
        state = CustomState(**values, time_step=time_step)
        return state


class PointMassNonlinearDynamics(VehicleDynamics):

    def __init__(self, vehicle_type: VehicleType):
        super(PointMassNonlinearDynamics, self).__init__(VehicleModel.PM, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        """
        Point Mass model dynamics function. Overrides the dynamics function of VehicleDynamics for PointMass model.

        :param t:
        :param x: state values, [position x, position y, velocity, orientation]
        :param u: input values, [acceleration, yaw rate]

        :return:
        """
        return [
            x[2] * np.cos(x[3]),
            x[2] * np.sin(x[3]),
            u[0],
            u[1]
        ]

    @property
    def input_bounds(self) -> Bounds:
        """
        Overrides the bounds method of Vehicle Model in order to return bounds for the Point Mass inputs.

        Bounds are
            - -max longitudinal acc <= acceleration <= max longitudinal acc
            - -max longitudinal acc <= acceleration_y <= max longitudinal acc

        :return: Bounds
        """
        return Bounds([-self.parameters.longitudinal.a_max, -self.parameters.longitudinal.a_max],
                      [self.parameters.longitudinal.a_max, self.parameters.longitudinal.a_max])

    def violates_friction_circle(self, x: Union[State, np.array], u: Union[State, np.array],
                                 throw: bool = False) -> bool:
        """
        Overrides the friction circle constraint method of Vehicle Model in order calculate
        friction circle constraint for the Point Mass model.

        :param x: current state
        :param u: the input which was used to simulate the next state
        :param throw: if set to false, will return bool instead of throwing exception (default=False)
        :return: True if the constraint was violated
        """
        u_vals = self.input_to_array(u)[0] if isinstance(u, State) else u
        x_vals = self.state_to_array(x)[0] if isinstance(x, State) else x

        a_long = u_vals[0]
        a_lat = u_vals[1] * x_vals[2]

        vals_power = a_long ** 2 + a_lat ** 2
        violates = vals_power > self.parameters.longitudinal.a_max ** 2

        if throw and violates:
            msg = f'Input violates friction circle constraint!\n' \
                  f'Init state: {x}\n\n Input:{u}'
            raise FrictionCircleException(msg)

        return violates

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0],
            state.position[1],
            state.velocity,
            state.orientation
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'velocity': x[2],
            'orientation': x[3]
        }
        return CustomState(**values, time_step=time_step)

    def _input_to_array(self, input: State) -> Tuple[np.array, int]:
        """ Overrides VehicleDynamics method. """
        values = [
            input.acceleration,
            input.yaw_rate,
        ]
        time_step = input.time_step
        return np.array(values), time_step

    def _array_to_input(self, u: np.array, time_step: int) -> State:
        """ Overrides VehicleDynamics method. """
        values = {
            'acceleration': u[0],
            'yaw_rate': u[1],
        }
        return CustomState(**values, time_step=time_step)

    def occupancy(self, x):
        """Compute the space occupied by the car, used only for the safety layer using reachable sets"""

        l = self.parameters.l
        w = self.parameters.w

        phi = x[3]
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        V = np.array([[l / 2, l / 2, -l / 2, -l / 2], [-w / 2, w / 2, w / 2, -w / 2]])

        V = np.dot(R, V) + np.reshape(x[0:2], (2, 1))

        return V


if __name__ == "__main__":
    continuous_vehicle = ContinuousVehicle()
