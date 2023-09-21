import numpy as np
from numpy.linalg import inv,norm
import matplotlib.pyplot as plt
from scipy.integrate import odeint


q_idx = 4
q_d_idx = 8
s_idx = 12
omega_idx = 15
omega_d_idx = 18
A_idx = 21

J_true = np.array([[20, 1.2, 0.9],[1.2, 17, 1.4],[0.9, 1.4, 15]])
J_hat_initial = 1.3*J_true

k_p = 2.125
k_w = 800
gamma = 10
e = np.exp(1)
pi = np.pi


def quat_to_matrix(q) :
    s = 1 / norm(q)**2 
    r_00 = 1 - 2*s*(q[2]**2 + q[3]**2) 
    r_01 = 2*s*(q[1]*q[2] - q[3]*q[0])
    r_02 = 2*s*(q[1]*q[3] + q[2]*q[0])
    r_10 = 2*s*(q[1]*q[2] + q[3]*q[0])
    r_11 = 1 - 2*s*(q[1]**2 + q[3]**2) 
    r_12 = 2*s*(q[2]*q[3] - q[1]*q[0])
    r_20 = 2*s*(q[1]*q[3] - q[2]*q[0])
    r_21 = 2*s*(q[2]*q[3] + q[1]*q[0])
    r_22 = 1 - 2*s*(q[1]**2 + q[2]**2)
    R = np.array([[r_00, r_01, r_02],[r_10,r_11,r_12],[r_20,r_21,r_22]])
    return R

def model(state, t) :
        q = state[:q_idx]
        R_q = quat_to_matrix(q)
        q_d = state[q_idx:q_d_idx]
        R_qd = quat_to_matrix(q_d)
        R_s = np.matmul(R_q, np.transpose(R_qd))
        s = state[q_d_idx:s_idx]
        omega = state[s_idx:omega_idx]
        omega_d = state[omega_idx:omega_d_idx]
        A_hat = state[omega_d_idx:A_idx]

        omega_d_dot = -0.3*np.sin(t)*(1-(e**(-0.01*(t**2)))) + 0.3*0.02*t*np.cos(t)*(e**(-0.01*(t**2))) + 0.006*t*np.sin(t)*(e**(-0.01*(t**2))) + (0.08*pi + 0.006*np.sin(t))*(e**(-0.01*(t**2))) - 0.02*(0.08*pi + 0.006*np.sin(t))*(t**2)*(e**(-0.01*(t**2)))
        domega_ddt = np.array([omega_d_dot, omega_d_dot, 0.])

        delta_omega = omega - np.matmul(R_s,omega_d)
        R_s_dot = -np.cross(delta_omega, R_s)
        phi = np.matmul(R_s_dot, omega_d) + np.matmul(R_s, domega_ddt)
        zeta = delta_omega + k_p*s[1:]
        A_hat_dot = gamma*zeta
        u =  - k_w*zeta + A_hat
        dqdt = quaternion_dynamics(q, omega)
        dq_ddt = quaternion_dynamics(q_d, omega_d)
        dsdt = quaternion_dynamics(s, delta_omega)
        domegadt = omega_dynamics(omega, u)
        derivatives = np.concatenate((dqdt, dq_ddt, dsdt, domegadt, domega_ddt, A_hat_dot))
        return derivatives

def quaternion_dynamics(q, omega) : # this function returns quaternion rates
        dq_0dt = -0.5*np.dot(q[1:],omega)
        dq_vdt = 0.5*(q[0]*omega + np.cross(q[1:],omega))
        quaternion_rates = np.transpose(np.array([dq_0dt, dq_vdt[0], dq_vdt[1], dq_vdt[2]]))
        return quaternion_rates

def omega_dynamics(omega, u): # this function returns angular velocity rates
        domegadt = np.matmul(inv(J_true), -np.cross(omega, np.matmul(J_true,omega)) + u)
        return domegadt

class SpaceCraft :
    def __init__(self, q_initial, q_d_initial, omega_initial, omega_d_initial, A_hat_initial):
        self.q = q_initial
        self.s = self.q
        self.q_d = q_d_initial
        self.omega = omega_initial
        self.omega_d = omega_d_initial
        self.A_hat = A_hat_initial # 3D vector
        self.dt = 0.1
        self.initial_state = np.concatenate((self.q, self.q_d , self.s, self.omega, self.omega_d, self.A_hat))
    

    def simulate(self, t_f) :
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.sol = odeint(model, self.initial_state, self.t, args=())
        self.q = []
        self.q_d = []
        self.s = []
        self.omega = []
        self.omega_d = []
        self.A_hat_norm = []
        self.u = []
        for sol in self.sol :
            q = sol[:q_idx]
            R_q = quat_to_matrix(q)
            q_d = sol[q_idx:q_d_idx]
            R_qd = quat_to_matrix(q_d)
            s = sol[q_d_idx:s_idx]
            R_s = np.matmul(R_q, np.transpose(R_qd))
            omega = sol[s_idx:omega_idx]
            omega_d = sol[omega_idx:omega_d_idx]
            delta_omega = omega - np.matmul(R_s,omega_d)
            zeta = delta_omega + k_p*s[1:]
            A_hat = sol[omega_d_idx:A_idx]
            u = -k_w*zeta + A_hat

            self.q.append(q)
            self.q_d.append(q_d)
            self.s.append(s)
            self.omega.append(omega)
            self.omega_d.append(omega_d)
            self.A_hat_norm.append(norm(A_hat))
            self.u.append(u)
        
    def plot(self):
        q = np.array(self.q)
        q_d = np.array(self.q_d)
        s = np.array(self.s)
        omega = np.array(self.omega)
        omega_d = np.array(self.omega_d)
        A_hat_norm = np.array(self.A_hat_norm)
        u = np.array(self.u)
        u_1 = u[:,0]
        u_2 = u[:,1]
        u_3 = u[:,2]
        q_0 = q[:,0]
        q_v_norm = norm(q[:,1:], axis=1)
        q_d0 = q_d[:,0]
        q_dv_norm = norm(q_d[:,1:], axis=1)
        s_0 = s[:,0]
        s_1 = s[:,1]
        s_2 = s[:,2]
        s_3 = s[:,3]
        s_v_norm = norm(s[:,1:], axis=1)
        omega_norm = norm(omega, axis=1)
        omega_d_norm = norm(omega_d, axis=1)

        plt.subplots(1)
        plt.plot(self.t, s_0, 'red', label=r"$s_0$")
        plt.plot(self.t, s_1, 'blue', label=r"$s_1$")
        plt.plot(self.t, s_2, 'green', label=r"$s_2$")
        plt.plot(self.t, s_3, 'magenta', label=r"$s_3$")
        plt.xlabel(r"$t$")
        plt.title(r"error quaternion $s$ vs time $t$")
        plt.legend()


        plt.subplots(1)
        plt.plot(self.t, u_1, 'cyan', label=r"$u_1$")
        plt.plot(self.t, u_2, 'red', label=r"$u_2$")
        plt.plot(self.t, u_3, 'black', label=r"$u_3$")
        plt.xlabel(r"$t$")
        plt.title(r"control inputs $u$ vs time $t$")
        plt.legend()

        # plt.subplots(1)
        # plt.plot(self.t, self.q_v_norm, 'red')
        # plt.xlabel("t")
        # plt.ylabel(r"$\vert \vert q_v \vert \vert$")
        # plt.title(r"$\vert \vert q_0\vert \vert$ vs time t")
        # plt.legend()

        plt.subplots(1)
        plt.plot(self.t, omega_norm, 'black', label=r"$\vert \vert \omega \vert \vert$")
        plt.plot(self.t, omega_d_norm, 'red', label=r"$\vert \vert \omega_d \vert \vert$")
        plt.xlabel("t")
        plt.ylabel(r"$\vert \vert \omega \vert \vert$")
        plt.title(r"$\vert \vert \omega \vert \vert$ vs time t")
        plt.legend()

        plt.subplots(1)
        plt.plot(self.t, A_hat_norm, 'blue', label=r"$\vert \vert \hat{A}\vert \vert$")
        plt.xlabel("t")
        plt.ylabel(r"$\vert \vert \hat{A} \vert \vert$")
        plt.title(r"$\vert \vert \hat{A} \vert \vert$ vs time t")
        plt.legend()

        plt.show()

if __name__ == '__main__' :
    q_0 = -np.sqrt(1 - (0.1826 * np.sqrt(3))**2)
    q_initial = np.transpose(np.array([q_0, 0.1826, 0.1826, 0.1826]))
    s_initial = q_initial
    R_q_initial = quat_to_matrix(q_initial)
    omega_initial = np.transpose(np.array([.1, .6, 1.]))
    q_d_initial = np.transpose(np.array([1., 0., 0., 0.]))
    R_q_d_initial = quat_to_matrix  (q_d_initial)
    omega_d_initial = np.transpose(np.array([0., 0., 1.]))
    omega_d_dot_initial = np.transpose(np.array([0.251327, 0.251327, 0.]))
    R_s_initial = np.matmul(R_q_initial,R_q_d_initial)
    delta_omega_initial = omega_initial - np.matmul(R_s_initial,omega_d_initial)
    R_s_dot_initial = -np.cross(delta_omega_initial, R_s_initial)
    phi_initial = np.matmul(R_s_dot_initial,omega_d_initial) + np.matmul(R_s_initial, omega_d_dot_initial)

    A_hat_initial = np.cross(omega_initial, np.matmul(J_hat_initial,omega_initial)) + np.matmul(J_hat_initial,phi_initial) - 0.5*k_p*np.matmul(J_hat_initial, (s_initial[0]*delta_omega_initial + np.cross(s_initial[1:],delta_omega_initial)))
    sc = SpaceCraft(q_initial, q_d_initial,omega_initial, omega_d_initial, A_hat_initial)
    sc.simulate(80)
    sc.plot()