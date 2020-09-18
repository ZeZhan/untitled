#!/usr/env python

import casadi as ca
import casadi.tools as ca_tools
import numpy as np
import time
from scipy import linalg
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

class MPC_test():
    def __init__(self, state_dim, dt=0.01, N=20):
        self.Ts = dt
        self.horizon = N
        self.g_ = 9.8066
        self.L = 0.15546
        self.d = 0.05
        self.mp = 0.073
        self.m = 1
        self.Ixx = 0.006816
        self.Iyy = 0.006802
        self.Izz = 0.00004548
        #数据来自Dynamics and command shaping control of quadcopters carring suspended loads
        # linear_drag_coefficient = [0.01, 0.01]

        # declare model variables
        ## control parameters
        roll_ref_ = ca.SX.sym('roll_ref_')
        pitch_ref_ = ca.SX.sym('pitch_ref_')
        yaw_ref_ = ca.SX.sym('yaw_ref_')
        thrust_ref_ = ca.SX.sym('thrust_ref_')
        controls_ = ca.vertcat(*[roll_ref_, pitch_ref_, thrust_ref_, yaw_ref_])
        num_controls = controls_.size()[0]
        ## control relevant parameters
        self.roll_tau = 0.257
        self.roll_gain = 0.75
        self.pitch_tau = 0.259
        self.pitch_gain = 0.78
        self.trajectory = None

        ## model states
        x_ = ca.SX.sym('x_')
        y_ = ca.SX.sym('y_')
        z_ = ca.SX.sym('z_')
        vx_ = ca.SX.sym('vx_')
        vy_ = ca.SX.sym('vy_')
        vz_ = ca.SX.sym('vz_')
        roll_ = ca.SX.sym('roll_')
        pitch_ = ca.SX.sym('pitch_')
        yaw_ = ca.SX.sym('yaw_')
        roll_p = ca.SX.sym('roll_p')
        pitch_p = ca.SX.sym('pitch_p')
        v_roll = ca.SX.sym('v_roll')
        v_pitch = ca.SX.sym('v_pitch')
        v_yaw = ca.SX.sym('v_yaw')
        v_roll_p = ca.SX.sym('v_roll_p')
        v_pitch_p = ca.SX.sym('v_pitch_p')
        states_ = ca.vertcat(*[x_, y_, z_, vx_, vy_, vz_, roll_, pitch_, yaw_, v_roll, v_pitch, v_yaw, roll_p, pitch_p, v_roll_p, v_pitch_p])
        num_states = states_.size()[0]
        ## external forces
        #ext_f_x_ = ca.SX.sym('ext_f_x_')
        #ext_f_y_ = ca.SX.sym('ext_f_y_')
        #ext_f_z_ = ca.SX.sym('ext_f_z_')
        #ext_f_ = ca.vertcat(*[ext_f_x_, ext_f_y_, ext_f_z_])

        ### this is one approach, one can also use directly self.dyn_function to define this work
        dragacc1_, dragacc2_ = self.aero_drag(states_, controls_)
        ext_f_x_, ext_f_y_, ext_f_z_ = self.ext_f(states_, controls_)
        #vx, vy, vz
        rhs = [states_[3], states_[4], states_[5]]
        #ax, ay, az
        rhs.append((ca.cos(roll_) * ca.cos(yaw_) * ca.sin(pitch_) + ca.sin(roll_) * ca.sin(yaw_)) * thrust_ref_ - dragacc1_ + ext_f_x_)
        rhs.append((ca.cos(roll_) * ca.sin(pitch_) * ca.sin(yaw_) - ca.cos(yaw_) * ca.sin(roll_)) * thrust_ref_ - dragacc2_ +ext_f_y_)
        rhs.append(-self.g_ + ca.cos(pitch_) * ca.cos(roll_) * thrust_ref_ + ext_f_z_)
        #v_roll, v_pitch, v_yaw
        rhs.append((self.roll_gain * roll_ref_ - roll_) / self.roll_tau)
        rhs.append((self.pitch_gain * pitch_ref_ - pitch_) / self.pitch_tau)
        rhs.append(0.0)
        #payload的三轴加速度
        a_x_p = states_[0]- (self.L * v_pitch_p * ca.sin(pitch_p) + self.L * ((v_roll* v_yaw)*(self.Izz- self.Ixx)/self.Iyy) * ca.cos(pitch_p) * ca.cos(roll_p) -
                      self.L * 0 * ca.sin(roll_p) * ca.cos(pitch_p) - 2 * self.L * v_pitch * v_pitch_p * ca.sin(pitch_p) * ca.cos(roll_p) -
                      2 * self.L * v_pitch * v_roll_p * ca.sin(roll_p) * ca.cos(pitch_p) - 2 * self.L * v_yaw * v_roll_p * ca.cos(pitch_p) *
                      ca.cos(roll_p) + 2 * self.L * v_yaw * v_pitch_p * ca.sin(roll_p) * ca.sin(pitch_p) - v_pitch * v_yaw * (self.Ixx - self.Iyy))
        a_y_p = states_[1]- (-self.L * v_roll_p **2 * ca.sin(roll_p)* ca.cos(pitch_p)- 2* self.L * v_roll_p* v_pitch_p* ca.cos(roll_p)* ca.sin(pitch_p) - v_pitch_p**2* ca.sin(roll_p)* ca.cos(pitch_p)+
                            self.L* ((v_pitch* v_yaw)*(self.Iyy- self.Izz)/self.Ixx)* ca.cos(pitch_p)*ca.cos(roll_p)-
                       + 2* self.L* v_yaw * v_pitch_p * ca.cos(pitch_p)- 2* self.L* v_pitch * ca.sin(pitch_p)* ca.cos(roll_p)- 2* self.L*
                      v_roll* v_roll_p * ca.sin(roll_p)* ca.cos(pitch_p)- v_roll* v_yaw* (self.Izz- self.Ixx))
        #a_z_p = states_[2]- (-self.L* v_pitch_p**2 * ca.cos(pitch_p)* ca.cos(roll_p)- self.L* v_roll_p**2 * ca.cos(roll_p)* ca.cos(pitch_p)+ 2* self.L* v_roll_p*
                     #v_pitch_p* ca.sin(pitch_p)* ca.sin(roll_p)+ self.L* ((v_pitch* v_yaw)*(self.Iyy- self.Izz)/self.Ixx)* ca.sin(roll_p)* ca.cos(pitch_p)+ self.L* ((v_roll* v_yaw)*(self.Izz- self.Ixx)/self.Iyy)*
                     #ca.sin(pitch_p)+ 2* self.L* v_roll* v_roll_p* ca.cos(pitch_p)*
                     #ca.cos(roll_p)- 2* self.L* v_pitch_p* v_roll* ca.sin(roll_p)* ca.sin(pitch_p)+ 2* self.L* v_pitch* v_pitch_p* ca.cos(pitch_p))+ v_pitch* v_roll* (self.Ixx- self.Iyy)
        #a_roll, a_pitch, a_yaw
        rhs.append((v_pitch* v_yaw)*(self.Iyy- self.Izz)/self.Ixx+ ca.cos(yaw_) * ca.cos(pitch_) * self.d * self.mp* (self.g_* ca.sin(roll_)* ca.cos(pitch_)+ a_y_p)/self.Ixx-
                   (ca.cos(yaw_)* ca.sin(pitch_)* ca.sin(roll_)- ca.sin(yaw_)* ca.cos(roll_))*self.d *self.mp* (a_x_p- self.g_* ca.sin(pitch_))/self.Ixx)
        rhs.append((v_roll* v_yaw)*(self.Izz- self.Ixx)/self.Iyy+ ca.sin(yaw_)* ca.cos(pitch_)* self.d* self.mp*(a_y_p+self.g_* ca.sin(roll_)* ca.cos(pitch_))/self.Iyy-(ca.sin(roll_)*
                    ca.sin(pitch_)* ca.sin(yaw_)+ ca.cos(yaw_)* ca.cos(roll_))* self.d* self.mp* (a_x_p- self.g_* ca.sin(pitch_))/self.Iyy)
        rhs.append(0.0)
        #v_roll_p, v_pitch_p (payload)
        rhs.append(states_[14])
        rhs.append(states_[15])
        #ax, az
        xx = (ca.cos(roll_) * ca.cos(yaw_) * ca.sin(pitch_) + ca.sin(roll_) * ca.sin(yaw_)) * thrust_ref_ - dragacc1_ + ext_f_x_
        yy = (ca.cos(roll_) * ca.sin(pitch_) * ca.sin(yaw_) - ca.cos(yaw_) * ca.sin(roll_)) * thrust_ref_ - dragacc2_ +ext_f_y_
        zz = -self.g_ + ca.cos(pitch_) * ca.cos(roll_) * thrust_ref_ + ext_f_z_
        #a_roll_p, a_pitch_p
        rhs.append(-ca.sin(roll_p) * ca.cos(roll_p) * v_pitch * v_pitch - ca.sin(roll_p) * ca.cos(roll_p)* v_pitch_p * v_pitch_p + ca.sin(roll_p) * ca.cos(roll_p)* (ca.sin(pitch_p)* v_roll_p - ca.cos(pitch_p)* v_yaw)**2
                   +2* (ca.cos(roll_p))**2 * v_pitch_p * (-ca.sin(pitch_p)* v_roll+ ca.cos(pitch_p)* v_yaw) + v_pitch*(2*ca.sin(roll_p)* ca.cos(roll_p)* v_pitch_p+ ((ca.cos(roll_p))**2- (ca.sin(roll_p))**2)* (ca.sin(pitch_p)* v_roll- ca.cos(pitch_p)* v_yaw))
                   -(ca.cos(pitch_p)*(dragacc2_- self.g_*ca.sin(roll_))+ (self.g_*(ca.sin(pitch_)* ca.sin(pitch_p)+ ca.cos(pitch_)* ca.cos(pitch_p))* ca.cos(roll_)+ dragacc1_* ca.sin(pitch_p))*ca.sin(roll_p)+
                   ca.cos(roll_p)* yy- ca.sin(roll_p)* (-ca.sin(pitch_p)*xx + ca.cos(pitch_p)* zz))/self.L+ ca.cos(pitch_p)* ((v_pitch* v_yaw)*(self.Iyy- self.Izz)/self.Ixx) + ca.sin(pitch_p)* 0)
        rhs.append(-ca.cos(pitch_p) * ca.sin(pitch_p) * v_roll * v_roll + 2* ca.tan(roll_p) * v_pitch_p * v_roll_p - 2 * ca.cos(pitch_p) * v_roll_p * v_yaw + ca.cos(pitch_p) * ca.sin(pitch_p) * v_yaw * v_yaw+
                   v_roll*(2* ca.sin(pitch_p)* v_roll_p +(ca.cos(pitch_p)* ca.cos(pitch_p)- ca.sin(pitch_p)* ca.sin(pitch_p))* v_yaw) + ca.tan(roll_p)* v_pitch *(ca.cos(pitch_p)* v_roll - 2* v_roll_p+ ca.sin(pitch_p)* v_yaw)+
                   (ca.cos(pitch_p)* (dragacc1_+ self.g_* ca.cos(roll_)* ca.sin(pitch_))+ (0- self.g_* ca.cos(pitch_)* ca.cos(roll_))* ca.sin(pitch_p)+ ca.cos(pitch_p)* xx+ ca.cos(roll_p)*
                   ca.cos(roll_p)* ca.sin(pitch_p)* zz)/self.L/ca.cos(roll_p)+ ((v_roll* v_yaw)*(self.Izz- self.Ixx)/self.Iyy) + ca.sin(pitch_p)* ca.tan(pitch_p)* ((v_pitch* v_yaw)*(self.Iyy- self.Izz)/self.Ixx)- ca.cos(pitch_p)*ca.tan(roll_p)*0)
        self.f = ca.Function('f', [states_, controls_], [ca.vertcat(*rhs)])



        ## additional parameters
        # self.external_forces = ca.SX.sym('F_ext', 3)
        self.Q_m = np.diag([80.0, 80.0, 120.0, 80.0, 80.0, 100.0, 80.0, 80.0, 120.0]) # position, velocity, roll, pitch, yaw
        # self.P_m = self.estimated_penalty_end_term()  # not working yet
        self.P_m = np.diag([86.21, 86.21, 120.95, 6.94, 6.94, 11.04]) # only p and v
        self.P_m[0, 3] = 6.45
        self.P_m[3, 0] = 6.45
        self.P_m[1, 4] = 6.45
        self.P_m[4, 1] = 6.45
        self.P_m[2, 5] = 10.95
        self.P_m[5, 2] = 10.95
        self.R_m = np.diag([50.0, 60.0, 50.0, 1.0]) # roll_ref, pitch_ref, thrust
        self.H_m = np.diag([80.0, 80.0])



        # MPC
        ## states and parameters
        U = ca.SX.sym('U', num_controls, self.horizon-1)
        X = ca.SX.sym('X', num_states, self.horizon)
        X_ref = ca.SX.sym('X_ref', num_states, self.horizon)
        F_ext = ca.SX.sym('F_ext', 3)
        ## constraints and cost
        ### end term
        obj = ca.mtimes([
            (X[:6, -1] - X_ref[:6, -1]).T,
            self.P_m,
            X[:6, -1] - X_ref[:6, -1]
        ])
        ### control cost
        for i in range(self.horizon-1):
            temp_ = ca.vertcat(U[:3, i], ca.cos(X[6, i])*ca.cos(X[7, i])*U[3, i] - self.g_)
            #print(temp_)
            obj = obj + ca.mtimes([
                temp_.T, self.R_m, temp_
            ])

        ### state cost
        for i in range(self.horizon-1):
            temp_ = X[:9, i] - X_ref[:9, i+1]
            obj = obj + ca.mtimes([temp_.T, self.Q_m, temp_])

        ## state cost of payload
        for i in range(self.horizon-1):
            temp_ = X[12:14, i] - X_ref[12:14, i + 1]
            obj = obj + ca.mtimes([temp_.T, self.H_m, temp_])

        ### constraints
        g = []
        g.append(X[:, 0]- X_ref[:, 0])
        for i in range(self.horizon-1):
            x_next_ = self.RK_4(X[:, i], U[:, i],F_ext)
            #print(x_next_)
            # x_next_ = self.dyn_function(X[:, i], U[:, i], F_ext) + X[:, i]
            g.append(X[:, i+1]-x_next_)

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))#输入变量
        #print(opt_variables)
        #问题所在
        opt_params = ca.vertcat(F_ext, ca.reshape(X_ref, -1, 1))#这里本来是opt_params = ca.vertcat(F_ext, ca.reshape(X_ref, -1, 1))，但是我我把外力用一个def来处理，类似drag的处理方式，那这边怎么办呢
        nlp_prob = {'f': obj, 'x':opt_variables, 'p':opt_params, 'g':ca.vertcat(*g)}

        opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.warm_start_init_point':'no'}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)


    def ext_f(self, states, controls):
        euler = states[6:9]
        euler_v = states[9:12]
        euler_p = states[12:14]
        euler_pv = states[14:16]

        #payload的三轴加速度
        a_x_p = states[0]- (self.L * euler_pv[1] * ca.sin(euler_p[1]) + self.L * ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy) * ca.cos(euler_p[1]) * ca.cos(euler_p[0]) -
                      self.L * 0 * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) - 2 * self.L * euler_v[1] * euler_pv[1] * ca.sin(euler_p[1]) * ca.cos(euler_p[0]) -
                      2 * self.L * euler_v[1] * euler_pv[0] * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) - 2 * self.L * euler_v[2] * euler_pv[0] * ca.cos(euler_p[1]) *
                      ca.cos(euler_p[0]) + 2 * self.L * euler_v[2] * euler_pv[1] * ca.sin(euler_p[0]) * ca.sin(euler_p[1]) - euler_v[1] * euler_v[2] * (self.Ixx - self.Iyy))

        a_y_p = states[1]- (-self.L * euler_pv[0] **2 * ca.sin(euler_p[0])* ca.cos(euler_p[1])- 2* self.L * euler_pv[0]* euler_pv[1]* ca.cos(euler_p[0])* ca.sin(euler_p[1]) - euler_pv[1]**2* ca.sin(euler_p[0])* ca.cos(euler_p[1])+
                            self.L* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)* ca.cos(euler_p[1])*ca.cos(euler_p[0])-
                       + 2* self.L* euler_v[2] * euler_pv[1] * ca.cos(euler_p[1])- 2* self.L* euler_v[1] * ca.sin(euler_p[1])* ca.cos(euler_p[0])- 2* self.L*
                      euler_v[0]* euler_pv[0] * ca.sin(euler_p[0])* ca.cos(euler_p[1])- euler_v[0]* euler_v[2]* (self.Izz- self.Ixx))

        a_z_p = states[2]- (-self.L* euler_pv[1] **2 * ca.cos(euler_p[1])* ca.cos(euler_p[0])- self.L* euler_pv[0]**2 * ca.cos(euler_p[0])* ca.cos(euler_p[1])+ 2* self.L* euler_pv[0]*
                     euler_pv[1]* ca.sin(euler_p[1])* ca.sin(euler_p[0])+ self.L* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)* ca.sin(euler_p[0])* ca.cos(euler_p[1])+ self.L* ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy)*
                     ca.sin(euler_p[1])+ 2* self.L* euler_v[0]* euler_pv[0]* ca.cos(euler_p[1])*
                     ca.cos(euler_p[0])- 2* self.L* euler_pv[1]* euler_v[0]* ca.sin(euler_p[0])* ca.sin(euler_p[1])+ 2* self.L* euler_v[1]* euler_pv[1]* ca.cos(euler_p[1]))+ euler_v[1]* euler_v[0]* (self.Ixx- self.Iyy)

        ext_f_x_ = (a_x_p- self.g_* ca.sin(euler[1]))* self.mp/ self.m
        ext_f_y_ = (a_y_p+ self.g_* ca.sin(euler[0])* ca.cos(euler[1]))*self.mp/self.m
        ext_f_z_ = (a_z_p+ self.g_* ca.cos(euler [1])* ca.cos(euler[0]))*self.mp/self.m
        #ext_f_x_ = (self.L * euler_pv[1] * ca.sin(euler_p[1]) + self.L * ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy) * ca.cos(euler_p[1]) * ca.cos(euler_p[0]) -
        #           2 * self.L * euler_v[1] * euler_pv[1] * ca.sin(euler_p[1]) * ca.cos(euler_p[0]) #- self.L * 0 * ca.sin(euler_p[0]) * ca.cos(euler_p[1])
        #           -2 * self.L * euler_v[1] * euler_pv[0] * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) #- 2 * self.L * euler_v[2] * euler_pv[0] * ca.cos(euler_p[1]) *ca.cos(euler_p[0])
        #            + 2 * self.L * euler_v[2] * euler_pv[1] * ca.sin(euler_p[0]) * ca.sin(euler_p[1]) + euler_v[1] * euler_v[2] *
        #              (self.Ixx - self.Iyy))* (self.mp/self.m)+ (self.mp/self.m) * self.g_ * ca.sin(euler[1])
        #ext_f_y_ = - (self.L * euler_pv[0]* euler_pv[0]* ca.sin(euler_p[0])* ca.cos(euler_p[1])- 2* self.L * euler_pv[0]* euler_pv[1]* ca.cos(euler_p[0])* ca.sin(euler_p[1]) -self.L* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)* ca.cos(euler_p[1])*ca.cos(euler_p[0])-
        #              self.L* euler_pv[1]* euler_pv[1]* ca.sin(euler_p[0])* ca.cos(euler_p[1]) - 2* self.L* euler_v[2]* euler_pv[1]* ca.cos(euler_p[1])+ 2* self.L* euler_v[0]* ca.sin(euler_p[1])* ca.cos(euler_p[0])+ 2* self.L*
        #              euler_v[0]* euler_pv[0]* ca.sin(euler_p[0])* ca.cos(euler_p[1])+ euler_v[0]* euler_v[2]* (self.Izz- self.Ixx))* self.mp/self.m - (self.mp/self.m) * self.g_ * ca.sin(euler[0])* ca.cos(euler[1])
        #ext_f_z_ = - self.g_ - self.mp/self.m* (self.L* euler_pv[1]* euler_pv[1]* ca.cos(euler_p[1])* ca.cos(euler_p[0])+ self.L* euler_pv[0]* euler_pv[0]* ca.cos(euler_p[0])* ca.cos(euler_p[1])- 2* self.L* euler_pv[0]*
        #             euler_pv[1]* ca.sin(euler_p[1])* ca.sin(euler_p[0])- self.L* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)* ca.sin(euler_p[0])* ca.cos(euler_p[1])- self.L* ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy)*
        #             ca.sin(euler_p[1])- 2* self.L* euler_v[0]* euler_pv[0]* ca.cos(euler_p[1])*
        #             ca.cos(euler_p[0])+ 2* self.L* euler_pv[1]* euler_v[0]* ca.sin(euler_p[0])* ca.sin(euler_p[1])- 2* self.L* euler_v[1]* euler_pv[1]* ca.cos(euler_p[1])+ euler_v[1]* euler_v[0]* (self.Ixx- self.Iyy)-
        #             self.g_ * ca.cos(euler[1])* ca.cos(euler[0]))
        return ext_f_x_, ext_f_y_, ext_f_z_



    def aero_drag(self, states, controls):
        # aero dynamic drag acceleration
        linear_drag_coefficient = [0.01, 0.01]
        thrust = controls[2]
        v = states[3:6]
        euler = states[6:9]
        dragacc1 = ca.sin(euler[1])*thrust*v[2] + ca.cos(euler[1])*ca.cos(euler[2])*linear_drag_coefficient[0]*thrust*v[0] - ca.cos(euler[1])*linear_drag_coefficient[1]*ca.sin(euler[2])*thrust*v[1]
        dragacc2 = (ca.cos(euler[0])*ca.sin(euler[2])-ca.cos(euler[2])*ca.sin(euler[0])*ca.sin(euler[1]))*linear_drag_coefficient[1]*thrust*v[0] - (ca.cos(euler[0])*ca.cos(euler[2]) + ca.sin(euler[1])*ca.sin(euler[0])*ca.sin(euler[2]))*linear_drag_coefficient[1]*thrust*v[1] - ca.cos(euler[1])*linear_drag_coefficient[1]*ca.sin(euler[0])*thrust*v[2]
        return dragacc1, dragacc2

    def dyn_function(self, states, controls, ext_f):
        euler = states[6:9]
        euler_v = states[9:12]
        euler_p = states[12:14]
        euler_pv = states[14:16]
        roll_ref = controls[0]
        pitch_ref = controls[1]
        thrust = controls[3]
        dragacc1, dragacc2 = self.aero_drag(states, controls)
        ext_f_x, ext_f_y, ext_f_z = self.ext_f(states, controls)
        # dynamic of the system
        rhs = [states[3], states[4], states[5]]
        rhs.append((ca.cos(euler[0])*ca.cos(euler[2])*ca.sin(euler[1])+ ca.sin(euler[0])*ca.sin(euler[2]))*thrust + ext_f_x + dragacc1+ ext_f[0])
        rhs.append((ca.cos(euler[0])*ca.sin(euler[1])*ca.sin(euler[2]) - ca.cos(euler[2])*ca.sin(euler[0]))*thrust - ext_f_y + dragacc2+ ext_f[1])
        rhs.append(-self.g_+ ca.cos(euler[0])* ca.cos(euler[1])* thrust - ext_f_z+ ext_f[2])
        # v_roll, v_pitch, v_yaw
        rhs.append((self.roll_gain * roll_ref- euler[0]) / self.roll_tau)
        rhs.append((self.pitch_gain * pitch_ref - euler[1]) / self.pitch_tau)
        rhs.append(0.0)
        #payload的三轴加速度
        a_x_p = states[0]- (self.L * euler_pv[1] * ca.sin(euler_p[1]) + self.L * ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy) * ca.cos(euler_p[1]) * ca.cos(euler_p[0]) -
                      self.L * 0 * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) - 2 * self.L * euler_v[1] * euler_pv[1] * ca.sin(euler_p[1]) * ca.cos(euler_p[0]) -
                      2 * self.L * euler_v[1] * euler_pv[0] * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) - 2 * self.L * euler_v[2] * euler_pv[0] * ca.cos(euler_p[1]) *
                      ca.cos(euler_p[0]) + 2 * self.L * euler_v[2] * euler_pv[1] * ca.sin(euler_p[0]) * ca.sin(euler_p[1]) - euler_v[1] * euler_v[2] * (self.Ixx - self.Iyy))
        a_y_p = states[1]- (-self.L * euler_pv[0] **2 * ca.sin(euler_p[0])* ca.cos(euler_p[1])- 2* self.L * euler_pv[0]* euler_pv[1]* ca.cos(euler_p[0])* ca.sin(euler_p[1]) - euler_pv[1]**2* ca.sin(euler_p[0])* ca.cos(euler_p[1])+
                            self.L* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)* ca.cos(euler_p[1])*ca.cos(euler_p[0])-
                       + 2* self.L* euler_v[2] * euler_pv[1] * ca.cos(euler_p[1])- 2* self.L* euler_v[1] * ca.sin(euler_p[1])* ca.cos(euler_p[0])- 2* self.L*
                      euler_v[0]* euler_pv[0] * ca.sin(euler_p[0])* ca.cos(euler_p[1])- euler_v[0]* euler_v[2]* (self.Izz- self.Ixx))
        #a_roll, a_pitch, a_yaw
        rhs.append((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx+ ca.cos(euler[2]) * ca.cos(euler[1]) * self.d * self.mp* (self.g_* ca.sin(euler[0])* ca.cos(euler[1])+
                 a_y_p)/self.Ixx- (ca.cos(euler[2])* ca.sin(euler[1])* ca.sin(euler[0])- ca.sin(euler[2])* ca.cos(euler[0]))*self.d *self.mp* (a_x_p- self.g_* ca.sin(euler[1]))/self.Ixx)
        rhs.append((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy+ ca.sin(euler[2])* ca.cos(euler[1])* self.d* self.mp*(a_y_p+ self.g_* ca.sin(euler[0])* ca.cos(euler[1]))/self.Iyy-(ca.sin(euler[0])*
                    ca.sin(euler[1])* ca.sin(euler[2])+ ca.cos(euler[2])* ca.cos(euler[0]))* self.d* self.mp* (a_x_p- self.g_* ca.sin(euler[1]))/self.Iyy)
        rhs.append(0.0)
        #v_roll_p, v_pitch_p (payload)
        rhs.append(states[14])
        rhs.append(states[15])
        #ax, az
        xx = (ca.cos(euler[0]) * ca.cos(euler[2]) * ca.sin(euler[1]) + ca.sin(euler[0]) * ca.sin(euler[2])) * thrust - dragacc1 + ext_f_x
        yy = (ca.cos(euler[0]) * ca.sin(euler[1]) * ca.sin(euler[2]) - ca.cos(euler[2]) * ca.sin(euler[0])) * thrust - dragacc2 +ext_f_y
        zz = -self.g_ + ca.cos(euler[1]) * ca.cos(euler[0]) * thrust + ext_f_z
        #a_roll_p, a_pitch_p
        rhs.append(-ca.sin(euler_p[0]) * ca.cos(euler_p[0]) * euler_v[1] * euler_v[1] - ca.sin(euler_p[0]) * ca.cos(euler_p[0])* euler_pv[1] * euler_pv[1] + ca.sin(euler_p[0]) * ca.cos(euler_p[0])* (ca.sin(euler_p[1])* euler_pv[0] - ca.cos(euler_p[1])* euler_v[2])** 2 +
                   2* ca.cos(euler_p[0]) * ca.cos(euler_p[0]) * euler_pv[1]* (-ca.sin(euler_p[1])* euler_v[0]+ ca.cos(euler_p[1])* euler_v[2])+ euler_v[1]*(2*ca.sin(euler_p[0])* ca.cos(euler_p[0])* euler_pv[1] + (ca.cos(euler_p[0])**2- ca.sin(euler_p[0])**2)*(ca.sin(euler_p[1])* euler_v[0]- ca.cos(euler_p[1])* euler_v[2]))-
                   (ca.cos(euler_p[1])*(dragacc2- self.g_*ca.sin(euler[0]))+ (self.g_*(ca.sin(euler[1])* ca.sin(euler_p[1])+ ca.cos(euler[1])* ca.cos(euler_p[1]))* ca.cos(euler[0])+ dragacc1* ca.sin(euler_p[1]))*ca.sin(euler_p[0])+
                   ca.cos(euler_p[0])* yy- ca.sin(euler_p[0])* (-ca.sin(euler_p[1])*xx + ca.cos(euler_p[1])* zz))/self.L+ ca.cos(euler_p[1])* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx) + ca.sin(euler_p[1])* 0)
        rhs.append(-ca.cos(euler_p[1]) * ca.sin(euler_p[1]) * euler_v[0] * euler_v[0] + 2* ca.tan(euler_p[0]) * euler_pv[1] * euler_pv[0] - 2 * ca.cos(euler_p[0]) * euler_pv[0] * euler_v[2] + ca.cos(euler_p[1]) * ca.sin(euler_p[1]) * euler_v[2] * euler_v[2]+
                   euler_v[0]* (2* ca.sin(euler_p[1])* euler_pv[0] +(ca.cos(euler_p[1])* ca.cos(euler_p[1])- ca.sin(euler_p[1])* ca.sin(euler_p[1]))* euler_v[2]) + ca.tan(euler_p[0])* euler_v[1] *(ca.cos(euler_p[1])* euler_v[0] - 2* euler_pv[0]+ ca.sin(euler_p[1])* euler_v[2])+
                   (ca.cos(euler_p[1])* (dragacc1+ self.g_* ca.cos(euler[0])* ca.sin(euler[1]))+ (0- self.g_* ca.cos(euler[1])* ca.cos(euler[0]))* ca.sin(euler_p[1])+ ca.cos(euler_p[1])* xx+ ca.cos(euler_p[0])*
                   ca.cos(euler_p[0])* ca.sin(euler_p[1])* zz)/self.L/ca.cos(euler_p[0])+ ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy) + ca.sin(euler_p[1])* ca.tan(euler_p[1])* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)- ca.cos(euler_p[1])*ca.tan(euler_p[0])*0)
        # rhs is in (X, 1) shape
        return ca.vertcat(*rhs)

    def dyn_np_function(self, states, controls, ext_f):
        euler = states[6:9]
        euler_v = states[9:12]
        euler_p = states[12:14]
        euler_pv = states[14:16]
        roll_ref = controls[0]
        pitch_ref = controls[1]
        thrust = controls[3]
        dragacc1, dragacc2 = self.aero_drag(states, controls)
        ext_f_x, ext_f_y, ext_f_z = self.ext_f(states, controls)
        # dynamic of the system
        rhs = states[3:6]
        rhs = np.concatenate((rhs, (np.cos(euler[0])*np.cos(euler[2])*np.sin(euler[1])+ np.sin(euler[0])*np.sin(euler[2]))*thrust + ext_f_x+ ext_f[0]))
        rhs = np.concatenate((rhs, (np.cos(euler[0])*np.sin(euler[1])*np.sin(euler[2]) - np.cos(euler[2])*np.sin(euler[0]))*thrust - ext_f_y+ ext_f[1]))
        rhs = np.concatenate((rhs, -self.g_+ np.cos(euler[0])* np.cos(euler[1])* thrust - ext_f_z+ ext_f[2]))
        # v_roll, v_pitch, v_yaw
        rhs = np.concatenate((rhs, np.array([(self.roll_gain* roll_ref- euler[0])/self.roll_tau])))
        rhs = np.concatenate((rhs, np.array([(self.pitch_gain* pitch_ref - euler[1])/self.pitch_tau])))
        rhs = np.concatenate((rhs, np.array([0.0])))
        #payload的三轴加速度
        a_x_p = states[0]- (self.L * euler_pv[1] * ca.sin(euler_p[1]) + self.L * ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy) * ca.cos(euler_p[1]) * ca.cos(euler_p[0]) -
                      self.L * 0 * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) - 2 * self.L * euler_v[1] * euler_pv[1] * ca.sin(euler_p[1]) * ca.cos(euler_p[0]) -
                      2 * self.L * euler_v[1] * euler_pv[0] * ca.sin(euler_p[0]) * ca.cos(euler_p[1]) - 2 * self.L * euler_v[2] * euler_pv[0] * ca.cos(euler_p[1]) *
                      ca.cos(euler_p[0]) + 2 * self.L * euler_v[2] * euler_pv[1] * ca.sin(euler_p[0]) * ca.sin(euler_p[1]) - euler_v[1] * euler_v[2] * (self.Ixx - self.Iyy))
        a_y_p = states[1]- (-self.L * euler_pv[0] **2 * ca.sin(euler_p[0])* ca.cos(euler_p[1])- 2* self.L * euler_pv[0]* euler_pv[1]* ca.cos(euler_p[0])* ca.sin(euler_p[1]) - euler_pv[1]**2* ca.sin(euler_p[0])* ca.cos(euler_p[1])+
                            self.L* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)* ca.cos(euler_p[1])*ca.cos(euler_p[0])-
                       + 2* self.L* euler_v[2] * euler_pv[1] * ca.cos(euler_p[1])- 2* self.L* euler_v[1] * ca.sin(euler_p[1])* ca.cos(euler_p[0])- 2* self.L*
                      euler_v[0]* euler_pv[0] * ca.sin(euler_p[0])* ca.cos(euler_p[1])- euler_v[0]* euler_v[2]* (self.Izz- self.Ixx))
        #a_roll, a_pitch, a_yaw
        rhs = np.concatenate((rhs, np.array([(euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx+ ca.cos(euler[2]) * ca.cos(euler[1]) * self.d * self.mp* (self.g_* ca.sin(euler[0])* ca.cos(euler[1])+
                 a_y_p)/self.Ixx- (ca.cos(euler[2])* ca.sin(euler[1])* ca.sin(euler[0])- ca.sin(euler[2])* ca.cos(euler[0]))*self.d *self.mp* (a_x_p- self.g_* ca.sin(euler[1]))/self.Ixx])))
        rhs = np.concatenate((rhs, np.array([(euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy+ ca.sin(euler[2])* ca.cos(euler[1])* self.d* self.mp*(a_y_p+ self.g_* ca.sin(euler[0])* ca.cos(euler[1]))/self.Iyy-(ca.sin(euler[0])*
                    ca.sin(euler[1])* ca.sin(euler[2])+ ca.cos(euler[2])* ca.cos(euler[0]))* self.d* self.mp* (a_x_p- self.g_* ca.sin(euler[1]))/self.Iyy])))
        rhs = np.concatenate((rhs, np.array([0.0])))
        # v_roll_p, v_pitch_p (payload)
        rhs = np.concatenate((rhs, np.array([states[14]])))
        rhs = np.concatenate((rhs, np.array([states[15]])))
        #ax, az
        xx = (ca.cos(euler[0]) * ca.cos(euler[2]) * ca.sin(euler[1]) + ca.sin(euler[0]) * ca.sin(euler[2])) * thrust - dragacc1 + ext_f_x
        yy = (ca.cos(euler[0]) * ca.sin(euler[1]) * ca.sin(euler[2]) - ca.cos(euler[2]) * ca.sin(euler[0])) * thrust - dragacc2 +ext_f_y
        zz = -self.g_ + ca.cos(euler[1]) * ca.cos(euler[0]) * thrust + ext_f_z
        #a_roll_p, a_pitch_p
        rhs = np.concatenate((rhs, np.array([-np.sin(euler_p[0]) * np.cos(euler_p[0]) * euler_v[1] * euler_v[1] - np.sin(euler_p[0]) * np.cos(euler_p[0])* euler_pv[1] * euler_pv[1] + np.sin(euler_p[0]) * np.cos(euler_p[0])* (np.sin(euler_p[1])* euler_pv[0] - np.cos(euler_p[1])* euler_v[2])** 2 +
                            2* np.cos(euler_p[0]) * np.cos(euler_p[0]) * euler_pv[1]* (-np.sin(euler_p[1])* euler_v[0]+ np.cos(euler_p[1])* euler_v[2])+
                        euler_v[1]*(2*np.sin(euler_p[0])* np.cos(euler_p[0])* euler_pv[1] +((np.cos(euler_p[0]))**2- (np.sin(euler_p[0]))**2)*(np.sin(euler_p[1])*euler_v[0]- np.cos(euler_p[1])* euler_v[2]))-
                   (np.cos(euler_p[1])*(dragacc2- self.g_*ca.sin(euler[0]))+ (self.g_*(np.sin(euler[1])* np.sin(euler_p[1])+ np.cos(euler[1])* np.cos(euler_p[1]))* np.cos(euler[0])+ dragacc1* np.sin(euler_p[1]))*np.sin(euler_p[0])+
                   ca.cos(euler_p[0])* yy- ca.sin(euler_p[0])* (-ca.sin(euler_p[1])*xx + ca.cos(euler_p[1])* zz))/self.L+ ca.cos(euler_p[1])* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx) + ca.sin(euler_p[1])* 0])))
        rhs = np.concatenate((rhs, np.array([(-np.cos(euler_p[1]) * np.sin(euler_p[1]) * euler_v[0] * euler_v[0] + 2* np.tan(euler_p[0]) * euler_pv[1] * euler_pv[0] - 2 * np.cos(euler_p[0]) * euler_pv[0] * euler_v[2] + np.cos(euler_p[1]) * np.sin(euler_p[1]) * euler_v[2] * euler_v[2]+
                   euler_v[0]* (2* np.sin(euler_p[1])* euler_pv[0] +(np.cos(euler_p[1])* np.cos(euler_p[1])- np.sin(euler_p[1])* np.sin(euler_p[1]))* euler_v[2]) + np.tan(euler_p[0])* euler_v[1] *(np.cos(euler_p[1])* euler_v[0] - 2* euler_pv[0]+ np.sin(euler_p[1])* euler_v[2])+
                   (np.cos(euler_p[1])* (dragacc1+ self.g_* np.cos(euler[0])* np.sin(euler[1]))+ (0- self.g_* np.cos(euler[1])* np.cos(euler[0]))* np.sin(euler_p[1])+ np.cos(euler_p[1])* xx+ np.cos(euler_p[0])*
                   np.cos(euler_p[0])* np.sin(euler_p[1])* zz)/self.L/np.cos(euler_p[0])+ ((euler_v[0]* euler_v[2])*(self.Izz- self.Ixx)/self.Iyy) + np.sin(euler_p[1])* np.tan(euler_p[1])* ((euler_v[1]* euler_v[2])*(self.Iyy- self.Izz)/self.Ixx)- np.cos(euler_p[1])*np.tan(euler_p[0])*0)])))
        # rhs is in (1, X) shape
        return rhs

    def RK_4(self, s_t_, c_, f_):
        # discretize Runge Kutta 4
        ## approach 1
        # k1 = self.f(s_t_, c_, f_)
        # k2 = self.f(s_t_+self.Ts/2.0*k1, c_, f_)
        # k3 = self.f(s_t_+self.Ts/2.0*k2, c_, f_)
        # k4 = self.f(s_t_+self.Ts*k3, c_, f_)
        ## approach 2
        k1 = self.dyn_function(s_t_, c_, f_)
        k2 = self.dyn_function(s_t_+self.Ts/2.0*k1, c_, f_)
        k3 = self.dyn_function(s_t_+self.Ts/2.0*k2, c_, f_)
        k4 = self.dyn_function(s_t_+self.Ts*k3, c_, f_)

        result_ = s_t_ + self.Ts/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        ## approach 3 using cov, working since integrate could not handle sx
        # result_ = self.f_cvo(x0=s_t_, p=ca.vcat([c_, f_]))['xf']

        return result_

    def model_based_movement(self, state, control, ext_F, t0, u_, x_):
        # print('state at t {0} is {1}'.format(t0, state))
        # print('control at t {0} is {1}'.format(t0, control))
        k1 = self.dyn_np_function(state, control, ext_F)
        k2 = self.dyn_np_function(state+self.Ts/2.0*k1.T, control, ext_F)
        k3 = self.dyn_np_function(state+self.Ts/2.0*k2.T, control, ext_F)
        k4 = self.dyn_np_function(state+self.Ts*k3.T, control, ext_F)
        x_next = state + self.Ts/6.0*(k1.T+2.0*k2.T+2.0*k3.T+k4.T)
        # nt_ = state + self.dyn_np_function(state, control, ext_F)*self.Ts
        # print('nt is {0}'.format(x_next))
        next_cmd_ = np.concatenate((u_[1:], u_[-1:]), axis=0)
        next_s_ = np.concatenate((x_[1:], x_[-1:]), axis=0)
        # print('next_cmd is {0}'.format(next_cmd_))
        # print('next_s is {0}'.format(next_s_))
        return t0+self.Ts, x_next, next_cmd_, next_s_

    def vertical_trajectory(self, current_state,):
        if current_state[2] >= self.trajectory[0, 2]:
            self.trajectory = np.concatenate((current_state.reshape(1, -1), self.trajectory[2:], self.trajectory[-1:]))
        return self.trajectory






if __name__ == '__main__':
    # model parameters
    n_states = 16
    N = 20
    n_controls = 4
    dt = 0.33
    L = 0.15546
    # create an MPC object
    mpc_obj = MPC_test(state_dim=n_states, dt=0.33, N=N)
    init_state = np.array([0.0]*n_states)
    current_state = init_state.copy()
    opt_commands = np.zeros((N-1, n_controls))
    next_states = np.zeros((N, n_states))
    # init_trajectory = np.array(
    #             [[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #              [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
    #             [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.1, 0.0, 0.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.2, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.2, 0.0, 0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.2, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.2, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.3, 0.0, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.3, .2, 0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.4, 0.2, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.5, 0.2, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.5, 0.2, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.7, 0.2, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.8, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.9, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [0.9, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #             ])
    init_trajectory = np.array(
                [[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ])
    mpc_obj.trajectory = init_trajectory.copy()
    next_trajectories = init_trajectory.copy()
    ext_forces = np.array([0.0, 0.0, 0]).reshape(-1, 1)
    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []
    for _ in range(N-1):
        lbx = lbx + [np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-45), 0.5*9.8066]
        ubx = ubx + [np.deg2rad(45), np.deg2rad(45), np.deg2rad(45), 1.5*9.8066]
    for _ in range(N):
        lbx = lbx + [-np.inf]*n_states
        ubx = ubx + [np.inf]*n_states
    # for saving data
    t0 = 0
    x_c = []
    u_c = []
    t_c = []
    x_states = []
    traj_c = []

    # start MPC
    sim_time = 600 # s
    mpc_iter = 0
    index_time = []
    start_time = time.time()
    #b = np.array([0.0]).reshape(-1, 1)
    while(mpc_iter < sim_time/dt and mpc_iter < 50):
        ## set parameters
        control_params = ca.vertcat(ext_forces.reshape(-1, 1), mpc_obj.trajectory.reshape(-1, 1))#mpc_obj.trajectory.reshape(-1)#这里本身是vertcat把力和轨迹弄成一维数组，我这样可以吗
        ## initial guess of the optimization targets
        init_control = ca.vertcat(opt_commands.reshape(-1, 1), next_states.reshape(-1, 1))
        ## solve the problem,求解器
        t_ = time.time()
        sol = mpc_obj.solver(x0=init_control, p=control_params, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        index_time.append(time.time() - t_)
        ## get results??
        estimated_opt = sol['x'].full()
        mpc_u_ = estimated_opt[:int(n_controls*(N-1))].reshape(N-1, n_controls)
        mpc_x_ = estimated_opt[int(n_controls*(N-1)):].reshape(N, n_states)
        # print(next_states)
        ## save results
        u_c.append(mpc_u_[0, :])
        t_c.append(t0)
        x_c.append(current_state)
        #x_states.append(mpc_x_)
        ## shift the movements, in the experiment, obtaining data from the
        ## the localization system
        #current state通过model_based_movement得到，也就是说输入是t0, mpc_u_[0, :], mpc_u_, mpc_x_
        t0, current_state, opt_commands, next_states = mpc_obj.model_based_movement(current_state, mpc_u_[0, :], ext_forces, t0, mpc_u_, mpc_x_)
        next_trajectories = mpc_obj.vertical_trajectory(current_state)
        traj_c.append(next_trajectories[1])
        # print(next_trajectories[:3])
        # print('current {}'.format(current_state))
        # print('control {}'.format(mpc_u_[0]))
        mpc_iter += 1
    print((time.time() - start_time)/mpc_iter)
    print(np.array(index_time).mean())
    print('max iter time {}'.format(np.max(index_time)))
    traj_s = np.array(x_c)
    traj_d = np.array(traj_c)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #蓝色实际飞行轨迹x_c用current state, 红色就是mpc推出的轨迹(将current states代入mpc,同时也给了参考轨迹)
    ax.plot(traj_s[:, 0], traj_s[:, 1], traj_s[:, 2], 'b')#x,y,z
    ax.plot(traj_d[:, 0], traj_d[:, 1], traj_d[:, 2], 'r')#x,y,z
    pickle.dump(ax, open('plot.pickle', 'wb'))
    ax = pickle.load(open('plot.pickle', 'rb'))
    #ax2 = fig.gca(projection='3d')
    #a = traj_s[:, 0]+ L
    #b = traj_s[:, 1]- L
    #c = traj_s[:, 2]- L
    #ax2.plot(a, b, c, 'k')
    #e = traj_d[:, 0]- L
    #f = traj_d[:, 1]- L
    #g = traj_d[:, 2]- L
    #ax2.plot(e, f, g, 'g')
    plt.show()
    #print(current_state)
    #print(traj_d)