import json
import numpy as np


class Optimizer:
    def __init__(self, num_devices, num_modules):
        self.num_devices = num_devices
        self.num_modules = num_modules
        self.num_flops = {}
        self.flop_speed = [0.0] * num_devices
        self.execution_time = np.zeros([num_devices, num_modules])
        self.ping_latency = np.zeros([num_devices, num_devices])
        self.bandwidths = np.zeros([num_devices, num_devices])
        self.m2m = {}
        self.m2m_size = None
        self.model_size = {}
        self.module_size = []
        self.total_mem = np.zeros([1, num_devices])
        self.ava_mem = np.zeros([1, num_devices])
        self.info_processed = False
        self.Solu = None
        self.Strategy = None

    def process_initial_info(self, num_flop: dict, flop_speed: list, ping_latency: np.ndarray,
                             bandwidths: np.ndarray, m2m: dict, model_size: dict,
                             total_mem, ava_mem):
        self.info_processed = True
        self.num_flops = num_flop
        self.flop_speed = flop_speed
        for i in range(len(self.flop_speed)):
            for k, val in self.num_flops.items():
                self.execution_time[i, k] = val / max(self.flop_speed[i], 1e-9)

        self.ping_latency = ping_latency
        self.bandwidths = bandwidths
        self.m2m = m2m
        self.m2m_size = np.zeros([len(self.m2m), len(self.m2m)])

        for i, res in self.m2m.items():
            print(f'res[]: {res}')
            for j, val in res['seq'].items():
                self.m2m_size[int(i)][int(j)] = sum(val) / 1_000_000
            for j, val in res['res'].items():
                self.m2m_size[int(i)][int(j)] = sum(val) / 1_000_000

        self.model_size = model_size
        for k, v in self.model_size.items():
            self.module_size.append(v["load"] / 1_000_000)

        self.total_mem = total_mem
        self.ava_mem = ava_mem

    def initial_module_arrangement(self):
        """
        Greedy consecutive-block assignment replacing the Gurobi ILP.
        Assigns modules to devices in roughly equal consecutive blocks,
        weighted by each device's compute speed (flop_speed).
        Returns a (num_devices x num_modules) binary numpy array.
        """
        assert self.info_processed, "Initial Optimization Information Must Be Processed!"
        D, M = self.num_devices, self.num_modules

        # Compute load-weighted share for each device
        total_speed = sum(max(s, 1e-9) for s in self.flop_speed)
        shares = [max(s, 1e-9) / total_speed for s in self.flop_speed]

        # Assign consecutive module blocks proportional to speed
        assignment = [[0] * M for _ in range(D)]
        start = 0
        for d in range(D):
            if d == D - 1:
                end = M
            else:
                end = start + max(1, round(shares[d] * M))
                end = min(end, M - (D - d - 1))  # leave at least 1 per remaining device
            for j in range(start, end):
                assignment[d][j] = 1
            start = end

        self.Solu = assignment
        print(f"\nGreedy module arrangement: {self.Solu}")
        return np.array(self.Solu)

    def dynamic_module_arrangement(self):
        """
        No-overlap version: same as initial arrangement.
        (Gurobi memory-overlap optimisation replaced with identity.)
        """
        assert self.Solu is not None, "Run initial_module_arrangement() first."
        self.Strategy = [row[:] for row in self.Solu]
        print(f"Dynamic module arrangement (no-overlap): {self.Strategy}")
        return self.Strategy
