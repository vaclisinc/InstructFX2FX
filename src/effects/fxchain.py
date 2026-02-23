import dasp_pytorch

class FXChain:
    """EQ + Compressor + Reverb chain."""

    def __init__(self, eq, compressor, reverb):
        self.eq = eq
        self.compressor = compressor
        self.reverb = reverb
        self.num_params = eq.num_params + compressor.num_params + reverb.num_params

    def __call__(self, audio, params):
        """Apply FX chain: audio [B,C,T], params [B, num_params]"""
        print(f"Applying FX chain with {params} parameters")

        eq_params = params[:, :self.eq.num_params]
        comp_params = params[:, self.eq.num_params:self.eq.num_params + self.compressor.num_params]
        reverb_params = params[:, self.eq.num_params + self.compressor.num_params:]

        x = self.eq.process_normalized(audio, eq_params)
        x = self.compressor.process_normalized(x, comp_params)
        # x = self.reverb.process_normalized(x, reverb_params)

        return x

class FXChainFactory:
    """Factory to create FX chains."""

    def create_fx_chain(self,sample_rate=44100, device='cpu'):
        """Create default FX chain."""
        eq = dasp_pytorch.ParametricEQ(sample_rate=sample_rate)
        comp = dasp_pytorch.Compressor(sample_rate=sample_rate)
        reverb = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)
        fx_chain = FXChain(eq, comp, reverb)
        print(f"✓ FX chain created: {fx_chain.num_params} parameters")
        return fx_chain