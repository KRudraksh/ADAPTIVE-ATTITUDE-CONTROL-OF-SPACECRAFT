import numpy as np
from numpy.linalg import inv,norm
import matplotlib.pyplot as plt
from scipy.integrate import odeint


q_idx = 4
q_d_idx = 8
s_idx = 12
omega_idx = 15
omega_d_idx = 18

J = np.array([[20, 1.2, 0.9],[1.2, 17, 1.4],[0.9, 1.4, 15]])

k_p = 40
k_w = 40



def quat_to_matrix(q) :
    s = 1 / np.linalg.norm(q)**2 
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
        domega_ddt = np.array([0., 0., 0.])

        delta_omega = omega - np.matmul(R_s,omega_d)
        R_s_dot = -np.cross(delta_omega, R_s)
        phi = np.matmul(R_s_dot, omega_d) + np.matmul(R_s, domega_ddt)
        u = -k_p*s[1:] - k_w*delta_omega + np.cross(omega, np.matmul(J,omega)) + np.matmul(J,phi) # let it stay uncommented if evaluating task 2
        # u = np.array([0., 0., 0.]) # uncomment for task 1
        dqdt = quaternion_dynamics(q, omega)
        dq_ddt = quaternion_dynamics(q_d, omega_d)
        dsdt = quaternion_dynamics(s, delta_omega)
        domegadt = omega_dynamics(omega, u)
        derivatives = np.concatenate((dqdt, dq_ddt, dsdt, domegadt, domega_ddt))
        return derivatives

def quaternion_dynamics(q, omega) : # this function returns quaternion rates
        dq_0dt = -0.5*np.dot(q[1:],omega)
        dq_vdt = 0.5*(q[0]*omega + np.cross(q[1:],omega))
        quaternion_rates = np.transpose(np.array([dq_0dt, dq_vdt[0], dq_vdt[1], dq_vdt[2]]))
        return quaternion_rates

def omega_dynamics(omega, u): # this function returns angular velocity rates
        domegadt = np.matmul(inv(J), -np.cross(omega, np.matmul(J,omega)) + u)
        return domegadt

class SpaceCraft :
    def __init__(self, q_initial, q_d_initial, omega_initial, omega_d_initial):
        self.q = q_initial
        self.s = self.q
        self.q_d = q_d_initial
        self.omega = omega_initial
        self.omega_d = omega_d_initial
        self.dt = 0.1
        self.initial_state = np.concatenate((self.q, self.q_d , self.s, self.omega, self.omega_d))
    

    def simulate(self, t_f) :
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.sol = odeint(model, self.initial_state, self.t, args=())
        self.omega_norm = []
        self.q_v_norm = []
        self.q_0 = []
        self.q_norm = []
        self.u = []
        for sol in self.sol :
            q = sol[:q_idx]
            q_d = sol[q_idx:q_d_idx]
            s = sol[q_d_idx:s_idx]
            omega = sol[s_idx:omega_idx]
            omega_d = sol[omega_idx:omega_d_idx]
            u = -k_p*s[1:] - k_w*omega + np.cross(omega,np.matmul(J,omega))
            self.omega_norm.append(norm(omega))
            self.q_v_norm.append(norm(q[1:]))
            self.q_0.append(q[0])
            self.q_norm.append(norm(q))
            self.u.append(u)
        
    def plot(self):
        u = np.array(self.u)
        u_1 = u[:,0]
        u_2 = u[:,1]
        u_3 = u[:,2]

        plt.subplots(1)
        plt.plot(self.t, u_1, 'blue', label=r"$u_1$")
        plt.plot(self.t, u_2, 'red', label=r"$u_2$")
        plt.plot(self.t, u_3, 'green', label=r"$u_3$")
        plt.xlabel(r"$t$")
        plt.title(r"control inputs $u$ vs time $t$")
        plt.legend()

        plt.subplots(1)
        plt.plot(self.t, self.q_0, 'black')
        plt.xlabel("t")
        plt.ylabel(r"$q_0$")
        plt.title(r"$q_0$ vs time t")
        plt.legend()

        plt.subplots(1)
        plt.plot(self.t, self.q_v_norm, 'red')
        plt.xlabel("t")
        plt.ylabel(r"$\vert \vert q_v \vert \vert$")
        plt.title(r"$\vert \vert q_v\vert \vert$ vs time t")
        plt.legend()

        plt.subplots(1)
        plt.plot(self.t, self.omega_norm, 'green')
        plt.xlabel("t")
        plt.ylabel(r"$\vert \vert \omega \vert \vert$")
        plt.title(r"$\vert \vert \omega \vert \vert$ vs time t")
        plt.legend()


        plt.show()

if __name__ == '__main__' :
    q_0 = -np.sqrt(1 - (0.1826 * np.sqrt(3))**2)
    q_initial = np.transpose(np.array([q_0, 0.1826, 0.1826, 0.1826]))
    omega_initial = np.transpose(np.array([0.1, 0.6, 1.0]))
    q_d_initial = np.transpose(np.array([1., 0., 0., 0.]))
    omega_d_initial = np.transpose(np.array([0., 0., 0]))
    sc = SpaceCraft(q_initial, q_d_initial,omega_initial, omega_d_initial)
    sc.simulate(60)
    sc.plot()