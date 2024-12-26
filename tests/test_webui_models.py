import os
import pytest
from unittest.mock import patch, MagicMock
from openai import OpenAI, OpenAIError, APIError
from webui.app import (
    get_configured_providers,
    load_available_models,
    filter_models,
    initialize_session_state
)

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'OPENROUTER_API_KEY': 'test-openrouter-key'
    }):
        yield

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(id='gpt-4'),
        MagicMock(id='gpt-3.5-turbo')
    ]
    return mock_response

@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(id='claude-3-opus-20240229'),
        MagicMock(id='claude-3-sonnet-20240229')
    ]
    return mock_response

def test_get_configured_providers(mock_env_vars):
    """Test that configured providers are correctly identified"""
    providers = get_configured_providers()
    assert 'openai' in providers
    assert 'anthropic' in providers
    assert 'openrouter' in providers
    
    # Test provider configurations
    assert providers['openai']['client_library'] is True
    assert providers['anthropic']['client_library'] is True
    assert providers['openrouter']['client_library'] is False

@patch('requests.get')
def test_load_available_models(mock_get, mock_env_vars, mock_openai_response, mock_anthropic_response):
    """Test loading available models from different providers"""
    # Mock OpenRouter API response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        'data': [
            {'id': 'openai/gpt-4-turbo'},
            {'id': 'anthropic/claude-3-opus'}
        ]
    }
    
    # Mock client library responses
    with patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value.models.list.return_value = mock_openai_response
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value.models.list.return_value = mock_anthropic_response
            
            models = load_available_models()
            
            # Check if all providers are present
            assert 'openai' in models
            assert 'anthropic' in models
            assert 'openrouter' in models
            
            # Check specific models
            assert 'openai/gpt-4' in models['openai']
            assert 'openai/gpt-3.5-turbo' in models['openai']
            assert 'anthropic/claude-3-opus-20240229' in models['anthropic']
            assert 'openai/gpt-4-turbo' in models['openrouter']

def test_filter_models():
    """Test model filtering functionality"""
    test_models = [
        'openai/gpt-4',
        'openai/gpt-3.5-turbo',
        'anthropic/claude-3-opus',
        'google/gemini-pro'
    ]
    
    # Test filtering by provider
    filtered = filter_models(test_models, 'openai')
    assert len(filtered) == 2
    assert all('openai' in model for model in filtered)
    
    # Test filtering by model name
    filtered = filter_models(test_models, 'gpt')
    assert len(filtered) == 2
    assert all('gpt' in model.lower() for model in filtered)
    
    # Test empty search query
    filtered = filter_models(test_models, '')
    assert len(filtered) == len(test_models)
    
    # Test no matches
    filtered = filter_models(test_models, 'nonexistent')
    assert len(filtered) == 0

class MockSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

@patch('streamlit.session_state', MockSessionState())
def test_initialize_session_state(mock_env_vars):
    """Test session state initialization"""
    with patch('webui.app.load_available_models') as mock_load_models:
        mock_load_models.return_value = {
            'openai': ['openai/gpt-4', 'openai/gpt-3.5-turbo'],
            'anthropic': ['anthropic/claude-3-opus']
        }
        
        initialize_session_state()
        
        from streamlit import session_state
        assert hasattr(session_state, 'messages')
        assert isinstance(session_state.messages, list)
        assert hasattr(session_state, 'connected')
        assert isinstance(session_state.connected, bool)
        assert hasattr(session_state, 'models')
        assert isinstance(session_state.models, dict)
        assert hasattr(session_state, 'websocket_thread_started')
        assert isinstance(session_state.websocket_thread_started, bool) 

def test_openai_model_id_validation():
    """Test validation of OpenAI model IDs"""
    test_models = [
        'openai/gpt-4o-mini',           # Invalid format
        'openai/gpt-4o-2024-11-20',     # Invalid format
        'gpt-4-turbo-preview',          # Valid format
        'gpt-3.5-turbo',                # Valid format
        'openai/invalid-model',         # Invalid model
        'gpt-4'                         # Valid format
    ]
    
    # Test filtering valid OpenAI models
    valid_models = [model for model in test_models if model in [
        'gpt-4-turbo-preview',
        'gpt-3.5-turbo',
        'gpt-4'
    ]]
    
    assert len(valid_models) == 3
    assert 'gpt-4-turbo-preview' in valid_models
    assert 'gpt-3.5-turbo' in valid_models
    assert 'gpt-4' in valid_models
    
    # Test invalid models are not included
    assert 'openai/gpt-4o-mini' not in valid_models
    assert 'openai/gpt-4o-2024-11-20' not in valid_models
    assert 'openai/invalid-model' not in valid_models

@patch('openai.OpenAI')
def test_openai_model_prefix_handling(mock_openai, mock_env_vars):
    """Test handling of OpenAI model ID prefixes"""
    # Setup mock response with valid models
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(id='gpt-4'),
        MagicMock(id='gpt-3.5-turbo')
    ]
    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Test model loading
    models = load_available_models()
    
    # Verify OpenAI models are properly formatted
    assert 'openai' in models
    openai_models = models['openai']
    
    # Check that models have correct prefix format
    assert 'openai/gpt-4' in openai_models
    assert 'openai/gpt-3.5-turbo' in openai_models
    
    # Verify no invalid model formats
    invalid_formats = [
        model for model in openai_models 
        if not model.startswith('openai/') or 
        model.replace('openai/', '') not in ['gpt-4', 'gpt-3.5-turbo']
    ]
    assert len(invalid_formats) == 0

@patch('openai.OpenAI')
def test_openai_invalid_model_format_error(mock_openai, mock_env_vars):
    """Test error handling for invalid OpenAI model formats"""
    # Setup mock error for invalid model format
    mock_client = MagicMock()
    mock_client.models.list.side_effect = OpenAIError("invalid model ID format")
    mock_openai.return_value = mock_client
    
    # Test error handling
    models = load_available_models()
    
    # Verify fallback to known good models
    assert 'openai' in models
    fallback_models = models['openai']
    assert set(fallback_models) == {
        'openai/gpt-4',
        'openai/gpt-3.5-turbo'
    } 