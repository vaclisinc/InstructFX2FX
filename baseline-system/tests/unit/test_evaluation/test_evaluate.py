"""Unit tests for pipeline evaluator and experiment tracking."""

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from judge_system.evaluation.evaluate import (
    PipelineEvaluator,
    generate_experiment_id,
    get_git_commit_hash,
)
from judge_system.evaluation.metrics import ExperimentMetrics
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest, ScoringResponse, ScoreDimension


class TestGenerateExperimentId:
    """Test suite for experiment ID generation."""

    def test_generates_valid_format(self):
        """Test that experiment ID has correct format."""
        exp_id = generate_experiment_id()

        # Check format: exp_{YYYYMMDD}_{HHMMSS}_{random}
        assert exp_id.startswith("exp_")
        parts = exp_id.split("_")
        assert len(parts) == 4
        assert parts[0] == "exp"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 4  # random suffix

    def test_generates_unique_ids(self):
        """Test that consecutive calls generate unique IDs."""
        id1 = generate_experiment_id()
        id2 = generate_experiment_id()

        assert id1 != id2

    def test_timestamp_is_valid(self):
        """Test that timestamp in ID is valid datetime format."""
        exp_id = generate_experiment_id()
        timestamp_part = exp_id.split("_")[1]

        # Should be parseable as YYYYMMDD
        datetime.strptime(timestamp_part, "%Y%m%d")


class TestGetGitCommitHash:
    """Test suite for git commit hash retrieval."""

    @patch("subprocess.run")
    def test_successful_git_hash_retrieval(self, mock_run):
        """Test successful retrieval of git commit hash."""
        mock_run.return_value = MagicMock(
            stdout="abc1234\n",
            returncode=0
        )

        result = get_git_commit_hash()

        assert result == "abc1234"
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

    @patch("subprocess.run")
    def test_git_command_failure(self, mock_run):
        """Test handling of git command failure."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        result = get_git_commit_hash()

        assert result is None

    @patch("subprocess.run")
    def test_git_not_installed(self, mock_run):
        """Test handling when git is not installed."""
        mock_run.side_effect = FileNotFoundError()

        result = get_git_commit_hash()

        assert result is None

    @patch("subprocess.run")
    def test_git_timeout(self, mock_run):
        """Test handling of git command timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("git", 5)

        result = get_git_commit_hash()

        assert result is None


class TestPipelineEvaluator:
    """Test suite for PipelineEvaluator class."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return str(tmp_path / "evaluator_output")

    @pytest.fixture
    def basic_config(self, temp_output_dir):
        """Create basic configuration."""
        return {
            "output_dir": temp_output_dir,
            "git_tracking": True,
        }

    @pytest.fixture
    def mock_scoring_system(self):
        """Create mock scoring system."""
        mock_system = MagicMock(spec=ScoringSystem)

        # Mock score_parameters to return a valid response
        async def mock_score_params(request):
            return ScoringResponse(
                overall_score=85.0,
                dimensions=[
                    ScoreDimension(
                        name="semantic_match",
                        score=90.0,
                        reasoning="Good semantic alignment"
                    ),
                    ScoreDimension(
                        name="technical_quality",
                        score=80.0,
                        reasoning="Decent technical quality"
                    ),
                ],
                feedback="Good overall match",
                suggestions=["Try adjusting reverb decay"],
                confidence=0.9
            )

        mock_system.score_parameters = AsyncMock(side_effect=mock_score_params)
        return mock_system

    @pytest.fixture
    def config_with_scoring(self, temp_output_dir, mock_scoring_system):
        """Create configuration with scoring system."""
        return {
            "output_dir": temp_output_dir,
            "git_tracking": True,
            "scoring_system": mock_scoring_system,
        }

    @pytest.fixture
    def evaluator(self, basic_config):
        """Create PipelineEvaluator instance."""
        return PipelineEvaluator(basic_config)

    @pytest.fixture
    def evaluator_with_scoring(self, config_with_scoring):
        """Create PipelineEvaluator instance with scoring."""
        return PipelineEvaluator(config_with_scoring)

    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        """Create sample audio file."""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_text("fake audio data")
        return str(audio_file)

    def test_initialization_with_valid_config(self, basic_config):
        """Test initialization with valid configuration."""
        evaluator = PipelineEvaluator(basic_config)

        assert evaluator.config == basic_config
        assert evaluator.output_dir == Path(basic_config["output_dir"])
        assert evaluator.git_tracking_enabled is True
        assert evaluator.scoring_system is None
        assert evaluator.metrics_collector is not None

    def test_initialization_creates_output_dir(self, basic_config):
        """Test that initialization creates output directory."""
        evaluator = PipelineEvaluator(basic_config)

        assert evaluator.output_dir.exists()
        assert evaluator.output_dir.is_dir()

    def test_initialization_without_output_dir_raises_error(self):
        """Test that missing output_dir raises ValueError."""
        config = {"git_tracking": True}

        with pytest.raises(ValueError, match="output_dir"):
            PipelineEvaluator(config)

    def test_initialization_with_scoring_system(self, config_with_scoring):
        """Test initialization with scoring system."""
        evaluator = PipelineEvaluator(config_with_scoring)

        assert evaluator.scoring_system is not None

    def test_git_tracking_disabled(self, temp_output_dir):
        """Test initialization with git tracking disabled."""
        config = {
            "output_dir": temp_output_dir,
            "git_tracking": False,
        }
        evaluator = PipelineEvaluator(config)

        assert evaluator.git_tracking_enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_single_with_parameters(
        self, evaluator_with_scoring, sample_audio_file
    ):
        """Test evaluate_single with provided parameters."""
        description = "warm reverb effect"
        parameters = {"reverb": {"decay": 0.5, "wet_dry": 0.3}}

        result = await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters
        )

        # Verify result structure
        assert isinstance(result, ExperimentMetrics)
        assert result.description == description
        assert result.parameters == parameters
        assert "overall_score" in result.scores
        assert result.scores["overall_score"] == 85.0
        assert result.scores["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_single_calls_scoring_system(
        self, evaluator_with_scoring, sample_audio_file, mock_scoring_system
    ):
        """Test that evaluate_single calls scoring system."""
        description = "bright eq boost"
        parameters = {"eq": {"freq": 2000, "gain": 3}}

        await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters
        )

        # Verify scoring system was called
        mock_scoring_system.score_parameters.assert_called_once()

        # Verify request structure
        call_args = mock_scoring_system.score_parameters.call_args[0][0]
        assert isinstance(call_args, ScoringRequest)
        assert call_args.description == description
        assert call_args.parameters == parameters

    @pytest.mark.asyncio
    async def test_evaluate_single_without_parameters_raises_error(
        self, evaluator, sample_audio_file
    ):
        """Test that evaluate_single without parameters raises NotImplementedError."""
        description = "test description"

        with pytest.raises(NotImplementedError, match="Parameter generation not yet integrated"):
            await evaluator.evaluate_single(
                description=description,
                audio_path=sample_audio_file,
                parameters=None
            )

    @pytest.mark.asyncio
    async def test_evaluate_single_with_empty_description_raises_error(
        self, evaluator, sample_audio_file
    ):
        """Test that empty description raises ValueError."""
        parameters = {"reverb": {"decay": 0.5}}

        with pytest.raises(ValueError, match="Description cannot be empty"):
            await evaluator.evaluate_single(
                description="",
                audio_path=sample_audio_file,
                parameters=parameters
            )

    @pytest.mark.asyncio
    async def test_evaluate_single_with_missing_audio_file_raises_error(
        self, evaluator
    ):
        """Test that missing audio file raises FileNotFoundError."""
        description = "test description"
        parameters = {"reverb": {"decay": 0.5}}

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await evaluator.evaluate_single(
                description=description,
                audio_path="/nonexistent/audio.wav",
                parameters=parameters
            )

    @pytest.mark.asyncio
    async def test_evaluate_single_tracks_experiment_id(
        self, evaluator_with_scoring, sample_audio_file
    ):
        """Test that evaluate_single generates unique experiment ID."""
        description = "test description"
        parameters = {"reverb": {"decay": 0.5}}

        result = await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters
        )

        assert result.experiment_id.startswith("exp_")

    @pytest.mark.asyncio
    @patch("judge_system.evaluation.evaluate.get_git_commit_hash")
    async def test_evaluate_single_tracks_git_hash(
        self, mock_git_hash, evaluator_with_scoring, sample_audio_file
    ):
        """Test that evaluate_single tracks git commit hash."""
        mock_git_hash.return_value = "abc1234"

        description = "test description"
        parameters = {"reverb": {"decay": 0.5}}

        result = await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters
        )

        assert result.metadata["git_hash"] == "abc1234"

    @pytest.mark.asyncio
    async def test_evaluate_single_with_custom_metadata(
        self, evaluator_with_scoring, sample_audio_file
    ):
        """Test evaluate_single with custom metadata."""
        description = "test description"
        parameters = {"reverb": {"decay": 0.5}}
        metadata = {"instrument": "guitar", "genre": "rock"}

        result = await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters,
            metadata=metadata
        )

        assert result.metadata["instrument"] == "guitar"
        assert result.metadata["genre"] == "rock"

    @pytest.mark.asyncio
    async def test_evaluate_single_collects_metrics(
        self, evaluator_with_scoring, sample_audio_file
    ):
        """Test that evaluate_single adds metrics to collector."""
        description = "test description"
        parameters = {"reverb": {"decay": 0.5}}

        await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters
        )

        assert evaluator_with_scoring.get_experiment_count() == 1

    @pytest.mark.asyncio
    async def test_evaluate_single_handles_scoring_error(
        self, evaluator_with_scoring, sample_audio_file, mock_scoring_system
    ):
        """Test that evaluate_single handles scoring errors gracefully."""
        # Make scoring raise an error
        mock_scoring_system.score_parameters.side_effect = Exception("Scoring failed")

        description = "test description"
        parameters = {"reverb": {"decay": 0.5}}

        result = await evaluator_with_scoring.evaluate_single(
            description=description,
            audio_path=sample_audio_file,
            parameters=parameters
        )

        # Should complete with error score rather than failing
        assert result.scores["error"] == -1.0

    @pytest.mark.asyncio
    async def test_evaluate_batch_with_valid_inputs(
        self, evaluator_with_scoring, tmp_path
    ):
        """Test evaluate_batch with valid inputs."""
        # Create multiple audio files
        audio_files = []
        for i in range(3):
            audio_file = tmp_path / f"audio_{i}.wav"
            audio_file.write_text(f"fake audio {i}")
            audio_files.append(str(audio_file))

        descriptions = [
            "warm reverb",
            "bright eq",
            "deep compression"
        ]

        parameters_list = [
            {"reverb": {"decay": 0.5}},
            {"eq": {"freq": 2000}},
            {"compressor": {"ratio": 4.0}},
        ]

        results = await evaluator_with_scoring.evaluate_batch(
            descriptions=descriptions,
            audio_paths=audio_files,
            parameters_list=parameters_list
        )

        assert len(results) == 3
        assert all(isinstance(r, ExperimentMetrics) for r in results)
        assert evaluator_with_scoring.get_experiment_count() == 3

    @pytest.mark.asyncio
    async def test_evaluate_batch_with_mismatched_lengths_raises_error(
        self, evaluator, sample_audio_file
    ):
        """Test that mismatched input lengths raise ValueError."""
        descriptions = ["desc1", "desc2"]
        audio_paths = [sample_audio_file]

        with pytest.raises(ValueError, match="Length mismatch"):
            await evaluator.evaluate_batch(
                descriptions=descriptions,
                audio_paths=audio_paths
            )

    @pytest.mark.asyncio
    async def test_evaluate_batch_with_mismatched_parameters_raises_error(
        self, evaluator, sample_audio_file
    ):
        """Test that mismatched parameters list length raises ValueError."""
        descriptions = ["desc1", "desc2"]
        audio_paths = [sample_audio_file, sample_audio_file]
        parameters_list = [{"reverb": {}}]

        with pytest.raises(ValueError, match="Length mismatch"):
            await evaluator.evaluate_batch(
                descriptions=descriptions,
                audio_paths=audio_paths,
                parameters_list=parameters_list
            )

    @pytest.mark.asyncio
    async def test_evaluate_batch_continues_on_individual_failure(
        self, evaluator_with_scoring, tmp_path
    ):
        """Test that evaluate_batch continues even if one sample fails."""
        # Create audio files (one will be invalid)
        audio1 = tmp_path / "audio1.wav"
        audio1.write_text("audio 1")

        descriptions = ["desc1", "desc2"]
        audio_paths = [str(audio1), "/nonexistent/audio.wav"]
        parameters_list = [{"reverb": {}}, {"eq": {}}]

        results = await evaluator_with_scoring.evaluate_batch(
            descriptions=descriptions,
            audio_paths=audio_paths,
            parameters_list=parameters_list
        )

        # Should have 1 successful result (the invalid one was skipped)
        assert len(results) == 1

    def test_generate_report_text_format(self, evaluator_with_scoring):
        """Test generate_report with text format."""
        # Manually add some metrics
        from judge_system.evaluation.metrics import ExperimentMetrics

        for i in range(3):
            metrics = ExperimentMetrics(
                experiment_id=f"exp_{i}",
                timestamp=datetime.now().isoformat(),
                description=f"test {i}",
                parameters={"test": {}},
                scores={"overall_score": 80.0 + i},
                audio_metrics={"loudness": -12.0 + i},
                metadata={}
            )
            evaluator_with_scoring.metrics_collector.collect(metrics)

        report = evaluator_with_scoring.generate_report(format="text")

        assert "EVALUATION REPORT" in report
        assert "Total Experiments: 3" in report
        assert "SCORE STATISTICS" in report
        assert "overall_score" in report

    def test_generate_report_markdown_format(self, evaluator_with_scoring):
        """Test generate_report with markdown format."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        metrics = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp=datetime.now().isoformat(),
            description="test",
            parameters={"test": {}},
            scores={"overall_score": 85.0},
            audio_metrics={"loudness": -12.0},
            metadata={}
        )
        evaluator_with_scoring.metrics_collector.collect(metrics)

        report = evaluator_with_scoring.generate_report(format="markdown")

        assert "# Evaluation Report" in report
        assert "**Total Experiments**:" in report
        assert "## Score Statistics" in report

    def test_generate_report_json_format(self, evaluator_with_scoring):
        """Test generate_report with JSON format."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        metrics = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp=datetime.now().isoformat(),
            description="test",
            parameters={"test": {}},
            scores={"overall_score": 85.0},
            audio_metrics={},
            metadata={}
        )
        evaluator_with_scoring.metrics_collector.collect(metrics)

        report = evaluator_with_scoring.generate_report(format="json")

        # Parse JSON
        data = json.loads(report)
        assert data["total_experiments"] == 1
        assert "statistics" in data
        assert "experiments" in data

    def test_generate_report_with_no_experiments_raises_error(self, evaluator):
        """Test that generate_report raises error with no experiments."""
        with pytest.raises(ValueError, match="No experiments collected"):
            evaluator.generate_report()

    def test_generate_report_with_invalid_format_raises_error(self, evaluator_with_scoring):
        """Test that invalid format raises ValueError."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        metrics = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp=datetime.now().isoformat(),
            description="test",
            parameters={},
            scores={},
            audio_metrics={},
            metadata={}
        )
        evaluator_with_scoring.metrics_collector.collect(metrics)

        with pytest.raises(ValueError, match="Invalid format"):
            evaluator_with_scoring.generate_report(format="xml")

    def test_export_results_json(self, evaluator_with_scoring):
        """Test export_results to JSON."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        metrics = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp=datetime.now().isoformat(),
            description="test",
            parameters={"reverb": {}},
            scores={"overall_score": 85.0},
            audio_metrics={},
            metadata={}
        )
        evaluator_with_scoring.metrics_collector.collect(metrics)

        result = evaluator_with_scoring.export_results()

        assert "json" in result
        assert Path(result["json"]).exists()

    def test_export_results_csv(self, evaluator_with_scoring, temp_output_dir):
        """Test export_results to CSV."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        metrics = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp=datetime.now().isoformat(),
            description="test",
            parameters={"reverb": {}},
            scores={"overall_score": 85.0},
            audio_metrics={},
            metadata={}
        )
        evaluator_with_scoring.metrics_collector.collect(metrics)

        csv_path = str(Path(temp_output_dir) / "test_export.csv")
        result = evaluator_with_scoring.export_results(csv_path=csv_path)

        assert "csv" in result
        assert Path(result["csv"]).exists()

    def test_export_results_with_no_experiments_raises_error(self, evaluator):
        """Test that export_results raises error with no experiments."""
        with pytest.raises(ValueError, match="No experiments collected"):
            evaluator.export_results()

    def test_clear_experiments(self, evaluator_with_scoring):
        """Test clear_experiments removes all collected data."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        metrics = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp=datetime.now().isoformat(),
            description="test",
            parameters={},
            scores={},
            audio_metrics={},
            metadata={}
        )
        evaluator_with_scoring.metrics_collector.collect(metrics)

        assert evaluator_with_scoring.get_experiment_count() == 1

        evaluator_with_scoring.clear_experiments()

        assert evaluator_with_scoring.get_experiment_count() == 0

    def test_get_experiment_count(self, evaluator_with_scoring):
        """Test get_experiment_count returns correct count."""
        from judge_system.evaluation.metrics import ExperimentMetrics

        assert evaluator_with_scoring.get_experiment_count() == 0

        for i in range(5):
            metrics = ExperimentMetrics(
                experiment_id=f"exp_{i}",
                timestamp=datetime.now().isoformat(),
                description="test",
                parameters={},
                scores={},
                audio_metrics={},
                metadata={}
            )
            evaluator_with_scoring.metrics_collector.collect(metrics)

        assert evaluator_with_scoring.get_experiment_count() == 5
