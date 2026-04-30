import torch
from agents.fx_selector import FXSelectorAgent
from agents.parser import Parser
from configurations.config import LossFunction
from session.session import Session

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Orchestrator:
    """Main entry point tying together FXSelectorAgent (Layer 1), Parser (Layer 2),
    and Session state management.

    Usage:
        session = Session(available_fx=["eq", "comp", "rev", "dist"])
        orch    = Orchestrator(llm_client, clap_model, device="cuda")

        # First prompt
        out = orch.run("bright", session, dry_audio)
        # Second prompt — eq already exists in session, routes to Case 1 or 3
        out = orch.run("warm", session, dry_audio)
    """

    def __init__(self, llm_client, clap_model, device=_DEFAULT_DEVICE, n_iterations=100, lr=0.01,
                 loss_function=LossFunction.SEMANTIC_SIMILARITY_LOSS):
        self.fx_selector = FXSelectorAgent(llm_client)
        self.parser      = Parser(llm_client, clap_model, device=device,
                                  n_iterations=n_iterations, lr=lr,
                                  loss_function=loss_function)

    def run(self, instruction: str, session: Session, audio) -> dict:
        """Process one user prompt end-to-end.

        Args:
            instruction: user descriptor word/phrase (e.g. "warm")
            session: Session object (updated in-place after each run)
            audio: torch tensor [1, T] at 44100 Hz

        Returns:
            {
                "fx_chain": ["eq", "rev"],               # full accumulated chain
                "params":   {"eq": {...}, "rev": {...}},  # optimized params for all FX
                "audio":    tensor,                       # rendered output audio
            }
        """
        selected_fx = self.fx_selector.select(instruction, session.available_fx)
        print(f"[FXSelector] '{instruction}' → {selected_fx}")

        # Carry forward all accumulated session FX, then append any new ones the
        # selector added. This ensures e.g. reverb from turn 1 stays in the chain
        # when distortion is added in turn 2.
        session_fx = list(session.current_params.keys())
        full_chain = session_fx + [fx for fx in selected_fx if fx not in session.current_params]
        if full_chain != selected_fx:
            print(f"[Orchestrator] Merged with session FX → {full_chain}")

        params, rendered_audio = self.parser.route(instruction, full_chain, session, audio)

        session.update(instruction, full_chain, params)
        print(f"[Orchestrator] Session updated. Current FX: {list(session.current_params.keys())}")

        return {"fx_chain": full_chain, "params": params, "audio": rendered_audio}
