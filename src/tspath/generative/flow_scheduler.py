import torch
from tspath.utils import sample_noise_like

__all__ = [
    "FlowScheduler",
    "CondOTScheduler",
]

class FlowScheduler:
    def __init__(self):
        pass
    
    def sample_xt(self, x0, x1, t, batch=None):
        raise NotImplementedError
    
    def sample_velocity(self, x0, x1, t, noise, batch=None):
        raise NotImplementedError
    
    def sample_conditional_path(self, x0, x1, t, batch):
        raise NotImplementedError
    
    def sample_time_and_conditional_path(self, x0, x1, batch):
        raise NotImplementedError
    
    @torch.no_grad()
    def sample(self, x0, num_steps, model, batch, conditioned=True, guidance_scale=0.0):
        raise NotImplementedError
    
    @torch.no_grad()
    def get_velocity(self, model, batch, conditioned=True, guidance_scale=0.0):
        if guidance_scale > 0.0:
            assert conditioned, "Classifier-free guidance requires conditional model."
        raise NotImplementedError


class CondOTScheduler(FlowScheduler):
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma
    
    def sample_xt(self, x0, x1, t, batch=None):
        
        # Get interpolated position
        xt = (1 - t) * x0 + t * x1
        noise = sample_noise_like(xt, batch)
        xt = xt + self.sigma * noise
        
        return xt, noise
    
    def sample_velocity(self, x0, x1, t, noise, batch=None):
        v_target = (x1 - x0)
        return v_target
    
    def sample_conditional_path(self, x0, x1, t, batch):
        # compute interpolated poisitions
        xt, noise = self.sample_xt(x0, x1, t, batch)
        
        # compute target velocity
        v_target = self.sample_velocity(x0, x1, t, noise, batch)
        
        return xt, v_target
    
    def sample_time_and_conditional_path(self, x0, x1, batch):
        
        # sample random time points
        batch_size = batch.max().item() + 1
        device = x0.device
        t = torch.rand((batch_size,1), device=device)[batch]
        
        # compute interpolated poisitions and target velocity
        xt, v_target = self.sample_conditional_path(x0, x1, t, batch)
        
        return xt, t, v_target
    
    
    @torch.no_grad()
    def sample(self, x0, num_steps, model, batch, conditioned=True, guidance_scale=0.0):
        batch_size = x0.size(0)
        x = x0.clone()
        trajectories = [x.clone()]
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step / num_steps, device=x.device)
            batch.pos_ts = x
            batch.t = t
            v = self.get_velocity(
                model, batch, conditioned=conditioned, guidance_scale=guidance_scale
            )
            x = x + v * (1.0 / num_steps)  # Euler integration step
            trajectories.append(x.clone())
        return x, trajectories
        
    
    @torch.no_grad()
    def get_velocity(self, model, batch, conditioned=True, guidance_scale=0.0):
        if guidance_scale > 0.0:
            assert conditioned, "Classifier-free guidance requires conditional model."
            # Get both conditional and unconditional predictions
            v_unconditioned = model(batch, conditioned=False)
            v_conditioned = model(batch, conditioned=True)
            
            # Combine them using classifier-free guidance
            v = v_unconditioned + guidance_scale * (v_conditioned - v_unconditioned)
        else:
            v = model(batch, conditioned=conditioned)
        
        return v
        
        
        
        
        