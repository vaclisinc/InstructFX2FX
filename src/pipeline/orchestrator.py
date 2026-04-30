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
        self.llm_client = llm_client
        self.clap_model = clap_model
        self.device = device
        self.n_iterations = n_iterations
        self.lr = lr
        self.loss_function = loss_function
        self.fx_selector = FXSelectorAgent(llm_client)
        self.parser      = Parser(llm_client, clap_model, device=device,
                                  n_iterations=n_iterations, lr=lr,
                                  loss_function=loss_function)

    def run(self, instruction: str, session: Session, audio) -> dict:
        return self._run_internal(instruction, session, audio, run_settings=None, return_metadata=False)

    def run_with_metadata(
        self,
        instruction: str,
        session: Session,
        audio,
        run_settings: dict | None = None,
    ) -> dict:
        """Variant of run() that exposes optimization metadata for API clients."""
        return self._run_internal(instruction, session, audio, run_settings=run_settings, return_metadata=True)

    def _make_parser(self, run_settings: dict | None = None) -> Parser:
        run_settings = run_settings or {}
        n_iterations = int(run_settings.get("n_iterations", self.n_iterations))
        lr = float(run_settings.get("learning_rate", self.lr))
        return Parser(
            self.llm_client,
            self.clap_model,
            device=self.device,
            n_iterations=n_iterations,
            lr=lr,
            loss_function=self.loss_function,
        )

    def _run_internal(
        self,
        instruction: str,
        session: Session,
        audio,
        run_settings: dict | None,
        return_metadata: bool,
    ) -> dict:
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
        llm_model = None
        if run_settings is not None:
            llm_model = run_settings.get("llm_model") or None

        selected_fx = self.fx_selector.select(
            instruction,
            session.available_fx,
            session.history,
            model=llm_model,
        )
        print(f"[FXSelector] '{instruction}' → {selected_fx}")

        # Carry forward all accumulated session FX, then append any new ones the
        # selector added. This ensures e.g. reverb from turn 1 stays in the chain
        # when distortion is added in turn 2.
        session_fx = list(session.current_params.keys())
        full_chain = session_fx + [fx for fx in selected_fx if fx not in session.current_params]
        if full_chain != selected_fx:
            print(f"[Orchestrator] Merged with session FX → {full_chain}")

        parser = self._make_parser(run_settings)
        if return_metadata:
            params, rendered_audio, metadata = parser.route(
                instruction,
                full_chain,
                session,
                audio,
                run_settings=run_settings,
                return_metadata=True,
            )
        else:
            params, rendered_audio = parser.route(
                instruction,
                full_chain,
                session,
                audio,
                run_settings=run_settings,
                return_metadata=False,
            )

        session.update(instruction, full_chain, params)
        print(f"[Orchestrator] Session updated. Current FX: {list(session.current_params.keys())}")

        result = {"fx_chain": full_chain, "params": params, "audio": rendered_audio}
        if return_metadata:
            result["metadata"] = metadata
        return result
