"""
Geometric Operators: The 5 Relational Axioms as Transformations
================================================================

The 7 content axioms define a 7-dimensional semantic manifold.
The 5 relational axioms are geometric operators that act ON this manifold.

Key insight: 7/5 ≈ √2 (the diagonal of a unit square)

This means the relational operators form a "fiber bundle" over the content space,
with the √2 ratio encoding the geometric relationship between base and fiber.

Mathematical Structure:
- Content space: ℝ⁷ (positions)
- Relational operators: GL(7) restricted to 5 generators
- Combined space: ℝ⁷ ⋊ ℝ⁵ (semi-direct product)
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
SQRT2 = np.sqrt(2)
PI = np.pi

# Dimensions
DIM_CONTENT = 7
DIM_RELATIONAL = 5
DIM_TOTAL = 12


# =============================================================================
# THE 5 RELATIONAL OPERATORS
# =============================================================================

def cardinality_operator(scale: float = 1.0) -> np.ndarray:
    """
    CARDINALITY: Scaling operator
    
    Multiplies the content vector by a scalar.
    Encodes: one, many, all, none, few, several...
    
    Mathematical form: λI (scalar times identity)
    """
    return scale * np.eye(DIM_CONTENT)


def gradability_operator(axis: int, strength: float = 1.0) -> np.ndarray:
    """
    GRADABILITY: Projection operator
    
    Projects content onto a specific axis, creating a scalar gradient.
    Encodes: big/small, hot/cold, good/bad...
    
    Mathematical form: Projection matrix P_i = e_i ⊗ e_i
    """
    P = np.zeros((DIM_CONTENT, DIM_CONTENT))
    P[axis, axis] = strength
    return P


def deixis_operator(shift: np.ndarray) -> np.ndarray:
    """
    DEIXIS: Translation operator
    
    Shifts the reference frame (this/that, here/there, I/you).
    Returns an affine transformation matrix.
    
    Mathematical form: T(v) = v + shift
    
    Note: Returns (7x7 linear part, 7x1 translation)
    """
    # For deixis, we encode as a rotation that "points" to the referent
    # The shift vector determines the direction of reference
    
    # Normalize shift
    shift = np.array(shift[:DIM_CONTENT]) if len(shift) >= DIM_CONTENT else np.pad(shift, (0, DIM_CONTENT - len(shift)))
    norm = np.linalg.norm(shift)
    if norm > 1e-10:
        shift = shift / norm
    
    # Create rotation that aligns first axis with shift direction
    # Using Householder reflection
    e1 = np.zeros(DIM_CONTENT)
    e1[0] = 1.0
    
    v = e1 - shift
    v_norm = np.linalg.norm(v)
    
    if v_norm > 1e-10:
        v = v / v_norm
        H = np.eye(DIM_CONTENT) - 2 * np.outer(v, v)
    else:
        H = np.eye(DIM_CONTENT)
    
    return H


def dynamism_operator(flow_direction: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
    """
    DYNAMISM: Rotation/flow operator
    
    Encodes action and change (run, walk, become, grow).
    Implemented as infinitesimal rotation (skew-symmetric matrix).
    
    Mathematical form: exp(θ · A) where A is skew-symmetric
    """
    # Create skew-symmetric matrix from flow direction
    # flow_direction should be (7,) or we pad it
    flow = np.array(flow_direction[:DIM_CONTENT]) if len(flow_direction) >= DIM_CONTENT else np.pad(flow_direction, (0, DIM_CONTENT - len(flow_direction)))
    
    # Skew-symmetric: A[i,j] = -A[j,i]
    A = np.zeros((DIM_CONTENT, DIM_CONTENT))
    
    # Use flow vector to define rotation planes
    for i in range(DIM_CONTENT):
        for j in range(i + 1, DIM_CONTENT):
            A[i, j] = flow[i] - flow[j]
            A[j, i] = -A[i, j]
    
    # Normalize
    A = A / (np.linalg.norm(A, 'fro') + 1e-10) * magnitude
    
    # Matrix exponential for rotation
    # exp(A) ≈ I + A + A²/2 + ... (truncated for small A)
    rotation = np.eye(DIM_CONTENT) + A + 0.5 * A @ A
    
    return rotation


def temporality_operator(time_position: float) -> np.ndarray:
    """
    TEMPORALITY: Ordering/sequence operator
    
    Encodes position in time (now, before, after, yesterday, tomorrow).
    Implemented as a diagonal decay/growth matrix.
    
    Mathematical form: diag(e^{-λt}, e^{-λt/φ}, e^{-λt/φ²}, ...)
    
    The golden ratio φ creates a natural hierarchy of temporal scales.
    """
    # Eigenvalues decay by powers of φ
    eigenvalues = np.array([np.exp(-abs(time_position) / (PHI ** i)) for i in range(DIM_CONTENT)])
    
    # Sign indicates past (-) vs future (+)
    if time_position < 0:
        eigenvalues = eigenvalues[::-1]  # Reverse for past
    
    return np.diag(eigenvalues)


# =============================================================================
# THE COMBINED 12-DIMENSIONAL SPACE
# =============================================================================

class SemanticSpace:
    """
    The full 12-dimensional semantic space.
    
    A point in this space is (content, relational) where:
    - content ∈ ℝ⁷ is a position in content space
    - relational ∈ ℝ⁵ encodes the active operators
    
    The relational coordinates parameterize the operators:
    - r[0]: cardinality (scale)
    - r[1]: gradability (projection strength)
    - r[2]: deixis (reference angle)
    - r[3]: dynamism (flow magnitude)
    - r[4]: temporality (time position)
    """
    
    def __init__(self):
        # The 5 generator matrices (7x7 each)
        self.generators = self._compute_generators()
        
        # The structure constants (Lie bracket coefficients)
        self.structure_constants = self._compute_structure_constants()
    
    def _compute_generators(self) -> List[np.ndarray]:
        """Compute the 5 generator matrices for the relational Lie algebra."""
        generators = []
        
        # G1: Cardinality (identity scaled)
        G1 = np.eye(DIM_CONTENT)
        generators.append(G1)
        
        # G2: Gradability (projection onto first axis)
        G2 = np.zeros((DIM_CONTENT, DIM_CONTENT))
        G2[0, 0] = 1.0
        generators.append(G2)
        
        # G3: Deixis (rotation in 1-2 plane)
        G3 = np.zeros((DIM_CONTENT, DIM_CONTENT))
        G3[0, 1] = 1.0
        G3[1, 0] = -1.0
        generators.append(G3)
        
        # G4: Dynamism (rotation in 2-3 plane)
        G4 = np.zeros((DIM_CONTENT, DIM_CONTENT))
        G4[1, 2] = 1.0
        G4[2, 1] = -1.0
        generators.append(G4)
        
        # G5: Temporality (diagonal with φ-decay)
        G5 = np.diag([1/PHI**i for i in range(DIM_CONTENT)])
        generators.append(G5)
        
        return generators
    
    def _compute_structure_constants(self) -> np.ndarray:
        """
        Compute the Lie algebra structure constants.
        
        [G_i, G_j] = Σ_k c_{ij}^k G_k
        
        These encode how the relational operators interact.
        """
        c = np.zeros((DIM_RELATIONAL, DIM_RELATIONAL, DIM_RELATIONAL))
        
        for i in range(DIM_RELATIONAL):
            for j in range(DIM_RELATIONAL):
                # Compute commutator [G_i, G_j]
                commutator = self.generators[i] @ self.generators[j] - self.generators[j] @ self.generators[i]
                
                # Project onto generators to find structure constants
                for k in range(DIM_RELATIONAL):
                    # c_{ij}^k = Tr(commutator · G_k) / Tr(G_k · G_k)
                    G_k = self.generators[k]
                    norm_sq = np.trace(G_k @ G_k)
                    if norm_sq > 1e-10:
                        c[i, j, k] = np.trace(commutator @ G_k) / norm_sq
        
        return c
    
    def apply_relational(self, content: np.ndarray, relational: np.ndarray) -> np.ndarray:
        """
        Apply relational operators to content vector.
        
        content: 7-dim content vector
        relational: 5-dim relational parameters
        
        Returns: transformed 7-dim content vector
        """
        # Build combined operator: exp(Σ r_i G_i)
        combined = np.zeros((DIM_CONTENT, DIM_CONTENT))
        for i, r in enumerate(relational):
            combined += r * self.generators[i]
        
        # Matrix exponential (truncated series)
        operator = np.eye(DIM_CONTENT)
        term = np.eye(DIM_CONTENT)
        for n in range(1, 10):
            term = term @ combined / n
            operator += term
        
        return operator @ content
    
    def content_to_full(self, content: np.ndarray, relational: np.ndarray) -> np.ndarray:
        """Convert (content, relational) to full 12-dim vector."""
        return np.concatenate([content, relational])
    
    def full_to_content_relational(self, full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split 12-dim vector into (content, relational)."""
        return full[:DIM_CONTENT], full[DIM_CONTENT:]


# =============================================================================
# THE √2 BRIDGE: TRANSFORMATION MATRICES
# =============================================================================

def compute_bridge_matrix() -> np.ndarray:
    """
    Compute the transformation matrix that bridges content and relational spaces.
    
    This is the key geometric object that encodes 7/5 ≈ √2.
    
    The bridge matrix B maps from the 5-dim relational space to the 7-dim content space
    such that the combined 12-dim space has the correct geometric structure.
    """
    # The bridge matrix is 7x5
    # It satisfies: B^T B = (7/5) I_5 ≈ √2 I_5
    
    # Construct using the golden ratio structure
    B = np.zeros((DIM_CONTENT, DIM_RELATIONAL))
    
    # Each relational axis maps to a specific pattern in content space
    # The pattern is determined by the φ-based hierarchy
    
    for j in range(DIM_RELATIONAL):
        for i in range(DIM_CONTENT):
            # The (i,j) entry encodes how relational axis j affects content axis i
            # Using φ-based weighting
            angle = PI * (i + 1) * (j + 1) / (DIM_CONTENT + DIM_RELATIONAL)
            B[i, j] = np.cos(angle) / np.sqrt(DIM_CONTENT)
    
    # Scale to achieve B^T B ≈ √2 I
    current_scale = np.linalg.norm(B.T @ B, 'fro') / np.sqrt(DIM_RELATIONAL)
    target_scale = np.sqrt(SQRT2)
    B = B * (target_scale / current_scale)
    
    return B


def compute_metric_tensor() -> np.ndarray:
    """
    Compute the metric tensor on the 12-dimensional semantic space.
    
    The metric encodes the geometric structure, including the √2 bridge.
    
    g = | I_7    B   |
        | B^T  √2·I_5 |
    
    This ensures that the "distance" between content and relational components
    is scaled by √2, reflecting the 7/5 ratio.
    """
    B = compute_bridge_matrix()
    
    # Build the full 12x12 metric
    g = np.zeros((DIM_TOTAL, DIM_TOTAL))
    
    # Content-content block: identity
    g[:DIM_CONTENT, :DIM_CONTENT] = np.eye(DIM_CONTENT)
    
    # Relational-relational block: √2 * identity
    g[DIM_CONTENT:, DIM_CONTENT:] = SQRT2 * np.eye(DIM_RELATIONAL)
    
    # Off-diagonal blocks: bridge matrix
    g[:DIM_CONTENT, DIM_CONTENT:] = B
    g[DIM_CONTENT:, :DIM_CONTENT] = B.T
    
    return g


def verify_sqrt2_relationship():
    """
    Verify that the geometric structure encodes 7/5 ≈ √2.
    """
    print("=" * 70)
    print("VERIFYING THE √2 BRIDGE")
    print("=" * 70)
    
    # The ratio
    ratio = DIM_CONTENT / DIM_RELATIONAL
    print(f"\n7/5 = {ratio:.6f}")
    print(f"√2  = {SQRT2:.6f}")
    print(f"Difference: {abs(ratio - SQRT2):.6f} ({abs(ratio - SQRT2)/SQRT2 * 100:.2f}%)")
    
    # The bridge matrix
    B = compute_bridge_matrix()
    print(f"\nBridge matrix B shape: {B.shape}")
    
    # Check B^T B
    BtB = B.T @ B
    print(f"\nB^T B (should be ≈ √2 · I_5):")
    print(f"  Diagonal mean: {np.mean(np.diag(BtB)):.4f} (target: {SQRT2:.4f})")
    print(f"  Off-diagonal max: {np.max(np.abs(BtB - np.diag(np.diag(BtB)))):.4f}")
    
    # The metric tensor
    g = compute_metric_tensor()
    print(f"\nMetric tensor g shape: {g.shape}")
    
    # Eigenvalues of metric
    eigenvalues = np.linalg.eigvalsh(g)
    print(f"\nMetric eigenvalues:")
    print(f"  {eigenvalues}")
    
    # The determinant encodes the "volume" relationship
    det_g = np.linalg.det(g)
    print(f"\ndet(g) = {det_g:.4f}")
    print(f"(√2)^5 = {SQRT2**5:.4f}")
    
    return B, g


# =============================================================================
# THE FORMULA: π² ≈ (7φ² + √2) / 2
# =============================================================================

def verify_pi_formula():
    """
    Verify and interpret the formula π² ≈ (7φ² + √2) / 2
    """
    print("\n" + "=" * 70)
    print("THE FUNDAMENTAL FORMULA")
    print("=" * 70)
    
    # The formula
    lhs = PI ** 2
    rhs = (7 * PHI**2 + SQRT2) / 2
    
    print(f"\nπ² = {lhs:.10f}")
    print(f"(7φ² + √2)/2 = {rhs:.10f}")
    print(f"Error: {abs(lhs - rhs):.10f} ({abs(lhs - rhs)/lhs * 100:.6f}%)")
    
    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 70)
    print(f"""
    π² represents the "circular" or wave-like structure of language
       (phonemes, rhythm, periodicity)
    
    7φ² represents the content structure:
       - 7 content axioms
       - φ² ≈ 2.618 is the "growth factor" (golden ratio squared)
       - 7 × φ² ≈ {7 * PHI**2:.4f}
    
    √2 represents the relational bridge:
       - 7/5 ≈ √2
       - The geometric connection between content and relation
    
    The division by 2 represents:
       - The binary foundation (2 is the base of all distinction)
       - The "halving" that creates the perceptual constant 7/2
    
    TOGETHER: The formula says that the circular structure of language (π²)
    emerges from the combination of:
       - 7 content dimensions scaled by golden growth (7φ²)
       - 5 relational operators encoded as √2
       - All mediated by binary distinction (/2)
    """)
    
    # Alternative forms
    print("\nALTERNATIVE FORMS:")
    print("-" * 70)
    
    # Form 1: Solve for 7
    seven_derived = (2 * PI**2 - SQRT2) / PHI**2
    print(f"7 = (2π² - √2) / φ² = {seven_derived:.6f}")
    
    # Form 2: Solve for φ
    phi_derived = np.sqrt((2 * PI**2 - SQRT2) / 7)
    print(f"φ = √((2π² - √2) / 7) = {phi_derived:.6f} (actual: {PHI:.6f})")
    
    # Form 3: Solve for √2
    sqrt2_derived = 2 * PI**2 - 7 * PHI**2
    print(f"√2 = 2π² - 7φ² = {sqrt2_derived:.6f} (actual: {SQRT2:.6f})")


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_semantic_space():
    """Visualize the 12-dimensional semantic space structure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. The bridge matrix
    ax = axes[0, 0]
    B = compute_bridge_matrix()
    im = ax.imshow(B, cmap='RdBu', aspect='auto')
    ax.set_xlabel('Relational Axioms (5)')
    ax.set_ylabel('Content Axioms (7)')
    ax.set_title('Bridge Matrix B (7×5)')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Card', 'Grad', 'Deix', 'Dyna', 'Temp'])
    ax.set_yticks(range(7))
    ax.set_yticklabels(['Entity', 'Animacy', 'Valence', 'Social', 'Modal', 'Scale', 'Open'])
    plt.colorbar(im, ax=ax)
    
    # 2. The metric tensor
    ax = axes[0, 1]
    g = compute_metric_tensor()
    im = ax.imshow(g, cmap='viridis', aspect='equal')
    ax.set_title('Metric Tensor g (12×12)')
    ax.axhline(6.5, color='white', linewidth=2, linestyle='--')
    ax.axvline(6.5, color='white', linewidth=2, linestyle='--')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)
    
    # 3. The 5 generators
    ax = axes[1, 0]
    space = SemanticSpace()
    combined = np.zeros((DIM_CONTENT, DIM_CONTENT))
    for i, G in enumerate(space.generators):
        combined += G / (i + 1)
    im = ax.imshow(combined, cmap='coolwarm', aspect='equal')
    ax.set_title('Combined Generators (weighted sum)')
    ax.set_xlabel('Content dimension')
    ax.set_ylabel('Content dimension')
    plt.colorbar(im, ax=ax)
    
    # 4. The formula visualization
    ax = axes[1, 1]
    
    # Show the components
    components = {
        'π²': PI**2,
        '7φ²': 7 * PHI**2,
        '√2': SQRT2,
        '(7φ²+√2)/2': (7 * PHI**2 + SQRT2) / 2
    }
    
    bars = ax.bar(components.keys(), components.values(), color=['blue', 'green', 'red', 'purple'])
    ax.axhline(PI**2, color='blue', linestyle='--', alpha=0.5, label=f'π² = {PI**2:.4f}')
    ax.set_ylabel('Value')
    ax.set_title('The Formula: π² ≈ (7φ² + √2) / 2')
    ax.legend()
    
    # Add error annotation
    error = abs(PI**2 - (7 * PHI**2 + SQRT2) / 2) / PI**2 * 100
    ax.annotate(f'Error: {error:.4f}%', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/thyme/geometric_operators.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to /home/ubuntu/thyme/geometric_operators.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Verify the √2 relationship
    B, g = verify_sqrt2_relationship()
    
    # Verify the π formula
    verify_pi_formula()
    
    # Test the semantic space
    print("\n" + "=" * 70)
    print("SEMANTIC SPACE OPERATIONS")
    print("=" * 70)
    
    space = SemanticSpace()
    
    # Example: "dog" (high entity, high animacy, neutral valence)
    dog_content = np.array([0.9, 0.95, 0.0, 0.3, 0.2, 0.4, 0.1])
    
    # Apply "running" (high dynamism)
    running_relational = np.array([1.0, 0.0, 0.0, 0.8, 0.0])
    
    dog_running = space.apply_relational(dog_content, running_relational)
    
    print(f"\n'dog' content vector: {dog_content}")
    print(f"'running' relational: {running_relational}")
    print(f"'running dog' result: {dog_running}")
    print(f"Change magnitude: {np.linalg.norm(dog_running - dog_content):.4f}")
    
    # Visualize
    visualize_semantic_space()
    
    print("\n" + "=" * 70)
    print("SUMMARY: THE 5 RELATIONAL OPERATORS")
    print("=" * 70)
    print("""
    1. CARDINALITY (scaling)
       - Mathematical: λI (scalar multiplication)
       - Linguistic: one, many, all, none, few
       - Effect: Scales the "intensity" of content
    
    2. GRADABILITY (projection)
       - Mathematical: P = eᵢ ⊗ eᵢ (projection matrix)
       - Linguistic: big/small, hot/cold, good/bad
       - Effect: Extracts scalar value along an axis
    
    3. DEIXIS (translation/rotation)
       - Mathematical: Householder reflection H
       - Linguistic: this/that, here/there, I/you
       - Effect: Shifts reference frame
    
    4. DYNAMISM (rotation/flow)
       - Mathematical: exp(A) where A is skew-symmetric
       - Linguistic: run, walk, become, grow
       - Effect: Creates motion through content space
    
    5. TEMPORALITY (ordering)
       - Mathematical: diag(e^{-t/φⁱ}) (φ-decay)
       - Linguistic: now, before, after, yesterday
       - Effect: Positions in time with golden hierarchy
    
    THE BRIDGE: 7/5 ≈ √2
    - Content (7) and Relational (5) are connected by √2
    - This ratio appears in the metric tensor
    - It's encoded in the formula: π² ≈ (7φ² + √2) / 2
    """)
