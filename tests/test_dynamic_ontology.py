import pytest
import os
import json
from unittest.mock import patch, MagicMock
from modules.module_05_ontology.graph import SkillOntologyGraph

@pytest.fixture
def ontology():
    return SkillOntologyGraph("backend_developer")

def test_adapt_to_candidate_no_api_key(ontology):
    """Test fallback when Ollama is unavailable."""
    # Since we test without a real Ollama running, it should naturally fallback
    # by catching the connection error.
    with patch('requests.post', side_effect=Exception("Connection refused")):
        success = ontology.adapt_to_candidate("JD", "Resume")
        
    assert success is False
    assert len(ontology.get_all_skills()) > 0
    assert "REST API" in ontology.get_all_skills()

@patch('requests.post')
def test_adapt_to_candidate_success(mock_post, ontology, monkeypatch):
    """Test successful dynamic adaptation using a mocked Ollama response."""
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": json.dumps({
            "nodes": ["MongoDB", "Express", "React", "NodeJS"],
            "edges": [["MongoDB", "NodeJS"], ["Express", "NodeJS"], ["React", "NodeJS"]]
        })
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    success = ontology.adapt_to_candidate("Need MERN stack developer", "I know MERN")
    
    assert success is True
    skills = ontology.get_all_skills()
    assert len(skills) == 4
    assert "MongoDB" in skills
    assert "REST API" not in skills  # The base skills were replaced
    
    prereqs = ontology.get_prerequisites("NodeJS")
    assert "MongoDB" in prereqs
    assert "Express" in prereqs

@patch('requests.post')
def test_adapt_to_candidate_invalid_json(mock_post, ontology):
    """Test that invalid JSON from LLM triggers fallback."""
    mock_response = MagicMock()
    # Invalid JSON missing edges
    mock_response.json.return_value = {
        "response": json.dumps({
            "nodes": ["MongoDB", "Express"]
        })
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    success = ontology.adapt_to_candidate("JD", "Resume")
    
    assert success is False
    assert "REST API" in ontology.get_all_skills() # Fell back to baseline

@patch('requests.post')
def test_adapt_to_candidate_markdown_stripping(mock_post, ontology):
    """Test that markdown code blocks are correctly stripped from LLM response."""
    mock_response = MagicMock()
    raw_json = json.dumps({
        "nodes": ["SkillA", "SkillB"],
        "edges": [["SkillA", "SkillB"]]
    })
    mock_response.json.return_value = {
        "response": f"```json\n{raw_json}\n```"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    success = ontology.adapt_to_candidate("JD", "Resume")
    
    assert success is True
    assert "SkillA" in ontology.get_all_skills()
