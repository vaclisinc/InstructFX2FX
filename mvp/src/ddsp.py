"""
Differentiable Digital Signal Processing (DDSP)

Provides differentiable audio effects using dasp_pytorch.
"""

import torch


class FXChain:
    """
    Wrapper for differentiable audio FX chain.

    Combines EQ, Compressor, and Reverb in a single chain.
    """

    def __init__(self, eq, compressor, reverb):
        """
        Initialize FX chain.

        Args:
            eq: Differentiable EQ module (e.g., dasp_pytorch.ParametricEQ)
            compressor: Differentiable compressor module
            reverb: Differentiable reverb module
        """
        self.eq = eq
        self.compressor = compressor
        self.reverb = reverb

        self.num_params = (
            self.eq.num_params +
            self.compressor.num_params +
            self.reverb.num_params
        )

    def __call__(self, audio: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Apply FX chain to audio.

        Args:
            audio: Input audio [B, C, T]
            params: FX parameters [B, num_params]

        Returns:
            Processed audio [B, C, T]
        """
        # Split params for each effect
        eq_params = params[:, :self.eq.num_params]
        comp_params = params[:, self.eq.num_params:self.eq.num_params + self.compressor.num_params]
        reverb_params = params[:, self.eq.num_params + self.compressor.num_params:]

        # Apply FX in sequence
        x = self.eq(audio, eq_params)
        x = self.compressor(x, comp_params)
        x = self.reverb(x, reverb_params)

        return x

    def get_param_names(self) -> list[str]:
        """Get names of all parameters."""
        names = []
        names.extend([f"eq_{i}" for i in range(self.eq.num_params)])
        names.extend([f"comp_{i}" for i in range(self.compressor.num_params)])
        names.extend([f"reverb_{i}" for i in range(self.reverb.num_params)])
        return names


def create_fx_chain(sample_rate: int = 44100, device='cpu') -> FXChain:
    """
    Create a default FX chain with EQ, Compressor, and Reverb.

    Args:
        sample_rate: Audio sample rate
        device: Device to create modules on

    Returns:
        FXChain instance
    """
    try:
        import dasp_pytorch
    except ImportError:
        raise ImportError(
            "dasp_pytorch not installed. Install with: "
            "pip install git+https://github.com/csteinmetz1/dasp-pytorch.git"
        )

    # Create differentiable FX modules
    eq = dasp_pytorch.ParametricEQ(
        sample_rate=sample_rate,
        num_bands=6  # 6-band parametric EQ (18 params)
    ).to(device)

    compressor = dasp_pytorch.Compressor(
        sample_rate=sample_rate
    ).to(device)

    reverb = dasp_pytorch.NoiseShapedReverb(
        sample_rate=sample_rate
    ).to(device)

    fx_chain = FXChain(eq, compressor, reverb)

    print(f"✓ FX chain created: {fx_chain.num_params} parameters")
    print(f"  - EQ: {eq.num_params} params")
    print(f"  - Compressor: {compressor.num_params} params")
    print(f"  - Reverb: {reverb.num_params} params")

    return fx_chain


def params_to_dict(params: torch.Tensor, fx_chain: FXChain) -> dict:
    """
    Convert parameter tensor to dictionary format.

    Args:
        params: Parameter tensor [B, num_params] or [num_params]
        fx_chain: FX chain to extract parameter structure from

    Returns:
        Dictionary with 'eq', 'compressor', 'reverb' keys
    """
    if params.ndim == 2:
        params = params.squeeze(0)

    params = params.cpu().numpy()

    # Split params
    eq_params = params[:fx_chain.eq.num_params]
    comp_start = fx_chain.eq.num_params
    comp_end = comp_start + fx_chain.compressor.num_params
    comp_params = params[comp_start:comp_end]
    reverb_params = params[comp_end:]

    return {
        'eq': eq_params.tolist(),
        'compressor': comp_params.tolist(),
        'reverb': reverb_params.tolist()
    }


def dict_to_params(params_dict: dict, fx_chain: FXChain, device='cpu') -> torch.Tensor:
    """
    Convert parameter dictionary to tensor format.

    Args:
        params_dict: Dictionary with 'eq', 'compressor', 'reverb' keys
        fx_chain: FX chain to determine parameter structure
        device: Device to create tensor on

    Returns:
        Parameter tensor [1, num_params]
    """
    # Concatenate all parameters
    all_params = []
    all_params.extend(params_dict.get('eq', []))
    all_params.extend(params_dict.get('compressor', []))
    all_params.extend(params_dict.get('reverb', []))

    # Convert to tensor
    params_tensor = torch.tensor(all_params, dtype=torch.float32, device=device)

    # Ensure correct shape
    if params_tensor.ndim == 1:
        params_tensor = params_tensor.unsqueeze(0)

    # Validate size
    expected_size = fx_chain.num_params
    if params_tensor.shape[1] != expected_size:
        raise ValueError(
            f"Parameter size mismatch: expected {expected_size}, got {params_tensor.shape[1]}"
        )

    return params_tensor
