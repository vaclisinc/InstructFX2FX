import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.generation.parameters import refine_loop

class TestText2FXIntegration:
    @pytest.fixture
    def mock_config(self):
        return {
            'refinement': {
                'method': 'text2fx',
                'max_iterations': 100,
                'learning_rate': 0.01
            },
            'llm': {'provider': 'test'},
            'prompts': {}
        }

    @patch('src.generation.parameters.refine_with_text2fx')
    def test_refine_loop_delegates_to_text2fx(self, mock_refine, mock_config):
        """Test that refine_loop calls refine_with_text2fx when method is 'text2fx'"""
        
        # Setup mock return
        mock_refine.return_value = {
            'best_params': {'eq': 'test'},
            'output_audio_path': 'output.wav',
            'history': []
        }
        
        user_prompt = "make it sound underwater"
        audio_path = "input.wav"
        
        result = refine_loop(user_prompt, audio_path, mock_config)
        
        # Verify call
        mock_refine.assert_called_once_with(user_prompt, audio_path, mock_config)
        assert result == mock_refine.return_value

    @patch('src.generation.parameters.generate_parameters')
    def test_refine_loop_defaults_to_llm(self, mock_generate, mock_config):
        """Test that refine_loop defaults to LLM logic if method is not text2fx"""
        
        # Change config to llm
        mock_config['refinement']['method'] = 'llm'
        mock_config['refinement']['max_iterations'] = 1 # Short loop
        
        # Mock internal calls to avoid full execution
        with patch('src.generation.parameters._generate_audio_description', return_value="desc"), \
             patch('src.generation.parameters.judge_audio', return_value=9.0):
            
            refine_loop("prompt", "audio.wav", mock_config)
            
        # Verify generate_parameters was called (start of LLM loop)
        mock_generate.assert_called()

if __name__ == "__main__":
    pytest.main([__file__])
