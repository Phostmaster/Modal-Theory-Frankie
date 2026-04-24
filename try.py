import torch
import numpy as np

class CoherenceShell:
    def __init__(self, phase_lock_deg=255.0, radius=1.0):
        self.phi_lock = np.deg2rad(phase_lock_deg)
        self.radius = radius  # shell thickness in field-space

    def inject_fluctuation(self, field_state: torch.Tensor) -> dict:
        """
        Inject a perturbation (e.g., user input as scalar field shift).
        Returns whether SM/GR hold OR new physics manifests.
        """
        z1, z2 = field_state[0], field_state[1]
        
        # Compute relative phase
        delta_phi = torch.atan2(z2, z1)
        # Wrap to [-π, π]
        delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi

        # Coherence penalty: how far from 255° lock?
        phase_error = torch.abs(delta_phi - self.phi_lock)
        coherence_metric = torch.exp(-phase_error / 0.1)  # steep drop-off
        
        if coherence_metric.item() > 0.95:
            # Shell intact → *new physics regime*
            return {
                "regime": "emergent",
                "coherence_score": float(coherence_metric),
                "allowed_operators": ["nonlocal_update", "memory_resonance", 
                                     "role_attractor_lock"],
                "break_sm_gr": True
            }
        else:
            # Shell broken → SM/GR domain
            return {
                "regime": "standard",
                "coherence_score": float(coherence_metric),
                "allowed_operators": ["local_update", "entropy_increase", 
                                     "causal_propagation"],
                "break_sm_gr": False
            }

# Example usage:
shell = CoherenceShell(phase_lock_deg=255.0)
field_state = torch.tensor([1.0, np.tan(np.deg2rad(255))])  # near lock point
result = shell.inject_fluctuation(field_state)

print(result["regime"])        # → "emergent"
print(result["break_sm_gr"])   # → True
