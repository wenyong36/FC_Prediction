import torch


class DMFNetwork:
    def __init__(self, w_ie, c_ij, g, **kwargs):
        assert isinstance(w_ie, torch.Tensor)
        assert isinstance(c_ij, torch.Tensor)
        self.m, self.n = w_ie.shape
        print('w_ie.shape, dtype', w_ie.shape, w_ie.dtype)
        self.w_ie = w_ie  # shape = m, n
        self.c_ij = c_ij  # shape = n, n
        self.I_e = torch.zeros_like(w_ie)  # shape = m, n
        self.I_i = torch.zeros_like(w_ie)  # shape = m, n
        self.s_e = torch.rand(w_ie.shape, dtype=w_ie.dtype)  # shape = m, n
        self.s_i = torch.rand(w_ie.shape, dtype=w_ie.dtype)  # shape = m, n
        self.r_e = torch.zeros_like(w_ie)  # shape = m, n
        self.r_i = torch.zeros_like(w_ie)  # shape = m, n

        self.g = kwargs.get("g", g)
        self.w_ee = kwargs.get("w_ee", 1.4)
        self.w_ei = kwargs.get("w_ei", 1.)
        self.w_ii = kwargs.get("w_ii", 1.)
        self.w_e = kwargs.get("w_e", 1.)
        self.w_i = kwargs.get("w_i", 0.7)
        self.j_nmda = kwargs.get("j_nmda", 0.15)
        self.j_i = kwargs.get("j_i", 1.)
        self.I_b = kwargs.get("I_b", 0.382)
        self.a_e = kwargs.get("a_e", 310)
        self.b_e = kwargs.get("b_e", 125)
        self.d_e = kwargs.get("d_e", 0.16)
        self.a_i = kwargs.get("a_i", 615)
        self.b_i = kwargs.get("b_i", 177)
        self.d_i = kwargs.get("d_i", 0.087)
        self.tau_e = kwargs.get("tau_e", 0.1)
        self.tau_i = kwargs.get("tau_i", 0.01)
        self.gamma = kwargs.get("gamma", 0.641)
        self.sigma = kwargs.get("sigma", 0.01)

    @staticmethod
    def phi(x, a, b, d):
        """
        activation function
        """
        return (a * x - b) / (1 - torch.exp(-d * (a * x - b)))

    def initialize(self, steps, dt, **kwargs):
        self.I_e = kwargs.get("r_e", self.I_e)
        self.I_i = kwargs.get("r_i", self.I_i)
        self.s_e = kwargs.get("s_e", self.s_e)
        self.s_i = kwargs.get("s_i", self.s_i)
        self.r_e = kwargs.get("I_e", self.r_e)
        self.r_i = kwargs.get("I_i", self.r_i)
        self.g = kwargs.get("g", self.g)
        self.run(steps=steps, dt=dt)

    def run(self, w_ie=None, steps=1, dt=1, print_info=True):
        if w_ie is not None:
            self.w_ie = w_ie
        for t in range(steps * dt):
            self.I_e = self.w_e * self.I_b + self.w_ee * self.j_nmda * self.s_i \
                       + self.g * self.j_nmda * torch.mm(self.s_e, self.c_ij.T) \
                       - self.j_i * (self.w_ie * self.s_i).sum(1)
            self.I_i = self.w_i * self.I_b + self.w_ei * self.j_nmda * self.s_e - self.w_ii * self.j_i * self.s_i
            self.r_e = self.phi(self.I_e, self.a_e, self.b_e, self.d_e)
            self.r_i = self.phi(self.I_i, self.a_i, self.b_i, self.d_i)
            self.s_e += dt * (-self.s_e / self.tau_e + self.gamma * (1 - self.s_e) * self.r_e +
                              self.sigma * torch.randn(self.s_e.shape, dtype=self.s_e.dtype))
            self.s_i += dt * (-self.s_i / self.tau_i + self.r_i +
                              self.sigma * torch.randn(self.s_i.shape, dtype=self.s_i.dtype))
        if print_info:
            print('re,ri,se,si mean', self.r_e.mean(), self.r_i.mean(), self.s_e.mean(), self.s_i.mean())
            print('se,si range', self.s_e.max(), self.s_e.min(), self.s_i.max(), self.s_i.min())
        return self.I_e, self.I_i, self.s_e, self.s_i, self.r_e, self.r_i

    def update(self, para, state):
        self.I_e = state[0]
        self.I_i = state[1]
        self.s_e = torch.clamp(state[2], 0, 1)
        self.s_i = torch.clamp(state[3], 0, 1)
        self.r_e = torch.clamp(state[4], 0, 200)
        self.r_i = torch.clamp(state[5], 0, 200)
        self.w_ie = para[0]
        self.g = para[1].mean()
