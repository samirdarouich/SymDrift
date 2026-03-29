import torch
from ase import Atoms
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data
from tspath.model import PaiNN
from ase.io import read

def ase_to_pyg(atoms: Atoms) -> Data:
    """Convert ASE Atoms object to PyG Data object."""
    x = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    batch = torch.zeros(x.size(0), dtype=torch.long)  # Single graph batch
    return Data(x=x, pos=pos, batch=batch)


def test_rotation_equivariance(atoms: Atoms, model, tolerance: float = 1e-5):
    """Test if model output is equivariant to rotations."""
    data = ase_to_pyg(atoms)

    # Original prediction
    with torch.no_grad():
        output_original = model(data)

    # Rotate atoms by random rotation matrix
    rotation = Rotation.random(random_state=42)
    rotated_pos = torch.tensor(
        rotation.apply(atoms.get_positions()), dtype=torch.float32
    )
    data_rotated = Data(x=data.x, pos=rotated_pos, batch=data.batch)

    with torch.no_grad():
        output_rotated = model(data_rotated)

    output_original_rotated = torch.tensor(rotation.apply(output_original), dtype=torch.float32)
    
    # For equivariant models, rotated output should match rotated original output
    assert torch.allclose(output_original_rotated, output_rotated, atol=tolerance), (
        "Model is not equivariant to rotations"
    )
    
    print("Rotation equivariance test passed!")


def test_permutation_equivariance(atoms: Atoms, model, tolerance: float = 1e-5):
    """Test if model output is equivariant to atom permutations."""
    data = ase_to_pyg(atoms)

    # Original prediction
    with torch.no_grad():
        output_original = model(data)

    # Permute atoms randomly
    perm_idx = torch.randperm(data.x.size(0))
    data_permuted = Data(x=data.x[perm_idx], pos=data.pos[perm_idx], batch=data.batch)

    with torch.no_grad():
        output_permuted = model(data_permuted)

    output_original_permuted = output_original[perm_idx]
    
    # For equivariant models, permuted output should match original
    assert torch.allclose(output_original_permuted, output_permuted, atol=tolerance), (
        "Model is not equivariant to permutations"
    )
    
    print("Permutation equivariance test passed!")
    
def test_translation_equivariance(atoms: Atoms, model, tolerance: float = 1e-5):
    """Test if model output is equivariant to translations."""
    data = ase_to_pyg(atoms)

    # Original prediction
    with torch.no_grad():
        output_original = model(data)

    # Translate atoms by random vector
    translation_vector = torch.rand(3) * 10.0  # Random translation up to 10 Å
    translated_pos = data.pos + translation_vector
    
    data_translated = Data(x=data.x, pos=translated_pos, batch=data.batch)
    with torch.no_grad():
        output_translated = model(data_translated)
    
    output_original_translated = output_original + translation_vector
    
    # For equivariant models, translated output should match original    
    assert torch.allclose(output_original_translated, output_translated, atol=tolerance), (
        "Model is not equivariant to translations"
    )
    print("Translation equivariance test passed!")
    
    
     
test_atoms = read("/home/samirdarouich/projects/TS_physics/tspath/data/geom_qm9/atoms/test.xyz", "10")

painn = PaiNN()


test_rotation_equivariance(test_atoms, painn)
test_permutation_equivariance(test_atoms, painn)
test_translation_equivariance(test_atoms, painn)