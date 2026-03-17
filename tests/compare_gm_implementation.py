import jax
import jax.numpy as jnp
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.descriptor import GaussianMomentDescriptor
from apax.nn.models import FeatureModel
import numpy as np

atomic_numbers = [6, 6, 6, 6, 6]  # 5-atom molecule (all carbon)
positions = np.array([[0.48886638, 0.18813352, 0.92679519],
       [0.37010578, 0.88013424, 0.99803836],
       [0.71482267, 0.60274513, 0.98626438],
       [0.99934985, 0.68662275, 0.79765171],
       [0.58943339, 0.61750176, 0.78739537]])

rf = RadialFunction(
    n_radial=3, emb_init=None, basis_fn=GaussianBasis(n_basis=2, r_max=10.0)
)
gm = GaussianMomentDescriptor(radial_fn=rf)
model = FeatureModel(representation=gm, readout=None)

positions_jax = jnp.asarray(positions)
Z = jnp.asarray(atomic_numbers)

edge_index = jnp.array([[1, 2, 0, 2, 3, 4], [0, 0, 1, 3, 4, 0]])  # simple ring connectivity

# Try APAX FeatureModel forward/init
try:
    params = model.init(jax.random.PRNGKey(0), R=positions_jax, Z=Z, neighbor=edge_index, box=np.array([0.0, 0.0, 0.0]), offsets=jnp.zeros_like(positions_jax))
    out = model.apply(params, R=positions_jax, Z=Z, neighbor=edge_index, box=np.array([0.0, 0.0, 0.0]), offsets=jnp.zeros_like(positions_jax))
    print("FeatureModel output:", out)
    print(out.shape)
except Exception as e:
    print("FeatureModel init/apply failed:", e)
    
    
import torch
from tspath.model import GaussianMomentDescriptor

atomic_numbers = torch.tensor([6, 6, 6, 6, 6])  # 5-atom molecule (all carbon)
positions = torch.tensor([[0.48886638, 0.18813352, 0.92679519],
       [0.37010578, 0.88013424, 0.99803836],
       [0.71482267, 0.60274513, 0.98626438],
       [0.99934985, 0.68662275, 0.79765171],
       [0.58943339, 0.61750176, 0.78739537]])

edge_index = torch.tensor([[1, 2, 0, 2, 3, 4], [0, 0, 1, 3, 4, 0]])
gm_descriptor = GaussianMomentDescriptor(
    n_radial=3, n_basis=2, max_radius=10.0, use_atom_type_embeddings=False
)

out = gm_descriptor(positions, edge_index, atomic_numbers)
print(out)