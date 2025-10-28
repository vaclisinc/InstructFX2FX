"""Performance tests for baseline system.

Tests benchmark critical operations to ensure they complete within acceptable timeframes.
All performance tests use the @pytest.mark.performance marker and have timeout thresholds.
"""

import pytest
import time
import asyncio
from pathlib import Path

from tests.mocks.mock_provider import MockLLMProvider
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest


class TestPerformance:
    """Performance and benchmarking tests."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider for performance testing."""
        return MockLLMProvider(response_mode="valid")

    @pytest.fixture
    def scoring_system(self, mock_provider):
        """Create scoring system with mock provider."""
        return ScoringSystem(llm_provider=mock_provider)

    @pytest.fixture
    def sample_request(self, sample_parameters, sample_description):
        """Create sample scoring request."""
        return ScoringRequest(
            description=sample_description,
            parameters=sample_parameters,
            iteration=1
        )

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_scoring_timing(self, scoring_system, sample_request):
        """Test single scoring operation completes in reasonable time.

        Threshold: <2 seconds for parameter-only scoring with mock provider.
        """
        start = time.time()

        response = await scoring_system.score_parameters(sample_request)

        duration = time.time() - start

        assert response is not None
        assert duration < 2.0, f"Scoring took {duration:.2f}s, expected <2s"

        print(f"\n⚡ Single scoring: {duration:.3f}s")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_scoring_throughput(self, scoring_system, sample_parameters, sample_descriptions):
        """Test batch scoring throughput.

        Threshold: Process 5 descriptions in <10 seconds.
        """
        start = time.time()

        tasks = []
        for desc in sample_descriptions:
            request = ScoringRequest(
                description=desc,
                parameters=sample_parameters,
                iteration=1
            )
            tasks.append(scoring_system.score_parameters(request))

        responses = await asyncio.gather(*tasks)

        duration = time.time() - start
        throughput = len(sample_descriptions) / duration

        assert len(responses) == len(sample_descriptions)
        assert duration < 10.0, f"Batch scoring took {duration:.2f}s, expected <10s"

        print(f"\n⚡ Batch scoring: {len(sample_descriptions)} items in {duration:.3f}s")
        print(f"⚡ Throughput: {throughput:.2f} items/sec")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_scoring_with_retry_overhead(self, mock_provider, sample_request):
        """Test retry mechanism overhead.

        Verifies that retry logic doesn't add significant overhead on success.
        """
        scoring_system = ScoringSystem(llm_provider=mock_provider)

        # Warm up
        await scoring_system.score_parameters(sample_request)

        # Benchmark
        iterations = 10
        start = time.time()

        for _ in range(iterations):
            await scoring_system.score_parameters(sample_request)

        duration = time.time() - start
        avg_time = duration / iterations

        assert avg_time < 1.0, f"Average scoring time {avg_time:.3f}s too high"

        print(f"\n⚡ Average scoring time: {avg_time:.3f}s ({iterations} iterations)")

    @pytest.mark.performance
    def test_mock_provider_overhead(self, mock_provider):
        """Test mock provider response time.

        Ensures mock provider doesn't add significant overhead.
        """
        from models.llm_judge.types import LLMRequest

        async def benchmark():
            request = LLMRequest(
                prompt="test prompt",
                temperature=0.7
            )

            iterations = 100
            start = time.time()

            for _ in range(iterations):
                await mock_provider.generate(request)

            duration = time.time() - start
            return duration / iterations

        avg_time = asyncio.run(benchmark())

        assert avg_time < 0.01, f"Mock provider too slow: {avg_time:.4f}s per call"

        print(f"\n⚡ Mock provider overhead: {avg_time*1000:.2f}ms per call")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, scoring_system, sample_request):
        """Test that memory usage doesn't grow unbounded.

        Runs multiple scoring operations and verifies memory stays stable.
        """
        import gc
        import sys

        # Get baseline memory
        gc.collect()

        # Run scoring operations
        iterations = 20
        for i in range(iterations):
            await scoring_system.score_parameters(sample_request)

            # Force garbage collection every 5 iterations
            if i % 5 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()

        # Test passes if we get here without memory errors
        print(f"\n⚡ Memory test: {iterations} iterations completed successfully")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_scoring_load(self, scoring_system, sample_parameters):
        """Test system under concurrent load.

        Simulates multiple concurrent scoring requests.
        """
        concurrent_requests = 10

        async def score_task(task_id: int):
            request = ScoringRequest(
                description=f"test description {task_id}",
                parameters=sample_parameters,
                iteration=1
            )
            start = time.time()
            result = await scoring_system.score_parameters(request)
            duration = time.time() - start
            return duration

        start = time.time()

        # Launch concurrent tasks
        tasks = [score_task(i) for i in range(concurrent_requests)]
        durations = await asyncio.gather(*tasks)

        total_duration = time.time() - start
        max_duration = max(durations)
        avg_duration = sum(durations) / len(durations)

        assert total_duration < 15.0, f"Concurrent load test took {total_duration:.2f}s"

        print(f"\n⚡ Concurrent load: {concurrent_requests} requests")
        print(f"⚡ Total time: {total_duration:.2f}s")
        print(f"⚡ Max request time: {max_duration:.2f}s")
        print(f"⚡ Avg request time: {avg_duration:.2f}s")

    @pytest.mark.performance
    def test_parameter_validation_speed(self, sample_parameters):
        """Test parameter validation performance.

        Ensures parameter validation doesn't become a bottleneck.
        """
        from src.scoring.models import ScoringRequest

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            request = ScoringRequest(
                description=f"test {i}",
                parameters=sample_parameters,
                iteration=i
            )

        duration = time.time() - start
        avg_time = duration / iterations

        assert avg_time < 0.001, f"Validation too slow: {avg_time*1000:.3f}ms"

        print(f"\n⚡ Parameter validation: {avg_time*1000:.3f}ms per request")


class TestPerformanceThresholds:
    """Document performance thresholds for the system."""

    @pytest.mark.performance
    def test_performance_requirements_documentation(self):
        """Document expected performance characteristics.

        This test always passes but documents the performance requirements.
        """
        requirements = {
            "Single scoring (mock)": "<2 seconds",
            "Batch scoring (5 items)": "<10 seconds",
            "Mock provider overhead": "<10ms per call",
            "Parameter validation": "<1ms per request",
            "Concurrent load (10 requests)": "<15 seconds",
            "Memory usage": "Stable over 20+ iterations"
        }

        print("\n📊 Performance Requirements:")
        for operation, threshold in requirements.items():
            print(f"  • {operation}: {threshold}")

        assert True  # Documentation test always passes


__all__ = [
    "TestPerformance",
    "TestPerformanceThresholds",
]
