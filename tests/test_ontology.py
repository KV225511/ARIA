import pytest
import os
import networkx as nx
from modules.module_05_ontology.graph import SkillOntologyGraph

def test_graph_initialization():
    """Test successful initialization with a valid role."""
    ontology = SkillOntologyGraph("backend_developer")
    assert len(ontology.get_all_skills()) > 0
    assert "REST API" in ontology.get_all_skills()

def test_invalid_role_initialization():
    """Test that an invalid role raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SkillOntologyGraph("nonexistent_role_12345")

def test_get_prerequisites_valid():
    """Test getting prerequisites for a known skill."""
    ontology = SkillOntologyGraph("backend_developer")
    prereqs = ontology.get_prerequisites("JWT")
    assert isinstance(prereqs, list)
    assert "Authentication" in prereqs

def test_get_prerequisites_invalid():
    """Test getting prerequisites for an unknown skill."""
    ontology = SkillOntologyGraph("backend_developer")
    prereqs = ontology.get_prerequisites("UnknownSkill")
    assert prereqs == []

def test_get_advanced_valid():
    """Test getting advanced topics for a known skill."""
    ontology = SkillOntologyGraph("backend_developer")
    advanced = ontology.get_advanced("JWT")
    assert isinstance(advanced, list)
    assert "Session Management" in advanced

def test_get_advanced_invalid():
    """Test getting advanced topics for an unknown skill."""
    ontology = SkillOntologyGraph("backend_developer")
    advanced = ontology.get_advanced("UnknownSkill")
    assert advanced == []
