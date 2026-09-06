import json
import os
import logging
import re
from dotenv import load_dotenv

from modules.module_05_ontology.grounding import (
    GroundedSkill,
    RoleProfile,
    build_role_profile,
    validate_role_profile,
)

# Configure logger
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SkillOntologyGraph:
    def __init__(self, role_name="backend_developer"):
        """
        Initializes the skill ontology graph by loading the given role JSON.
        """
        self.role_name = role_name
        # Internal adjacency list: skill -> set of advanced skills
        self.successors = {}
        # Internal adjacency list: skill -> set of prerequisite skills
        self.predecessors = {}
        self.nodes = set()
        
        load_dotenv()
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.api_endpoint = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate"
        
        self.inferred_role = role_name
        self.inferred_experience = "Mid-Level"
        self.role_profile: RoleProfile | None = None
        self.skill_metadata: dict[str, GroundedSkill] = {}
        
        self._load_graph()
        
    def _add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.successors[node] = set()
            self.predecessors[node] = set()
            
    def _add_edge(self, src, dst):
        self._add_node(src)
        self._add_node(dst)
        self.successors[src].add(dst)
        self.predecessors[dst].add(src)
        
    def _clear_graph(self):
        self.nodes.clear()
        self.successors.clear()
        self.predecessors.clear()

    def _load_graph(self):
        json_path = os.path.join(BASE_DIR, "roles", f"{self.role_name}.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Role file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self._clear_graph()
        for node in data.get("nodes", []):
            self._add_node(node)
        for edge in data.get("edges", []):
            if len(edge) == 2:
                self._add_edge(edge[0], edge[1])
                
        self.base_data = data # Store base data for dynamic adaptation
        
    def adapt_to_candidate(self, jd_text: str, resume_text: str) -> bool:
        """
        Takes JD and Resume text, uses local LLM to dynamically add missing skills.
        Also infers the target role and experience level.
        Returns True if successful, False if fell back to baseline.
        """
        logger.info("Building evidence-backed ontology from the complete JD...")
        try:
            profile = build_role_profile(jd_text, resume_text)
            validate_role_profile(profile, jd_text)
            self._clear_graph()
            self.role_profile = profile
            self.skill_metadata = {
                skill.canonical_name: skill for skill in profile.skills
            }
            names_by_id = {
                skill.skill_id: skill.canonical_name for skill in profile.skills
            }
            for skill in profile.skills:
                self._add_node(skill.canonical_name)
            for skill in profile.skills:
                for prerequisite_id in skill.prerequisite_ids:
                    prerequisite = names_by_id.get(prerequisite_id)
                    if prerequisite:
                        self._add_edge(prerequisite, skill.canonical_name)
            self._validate_graph()
            self.inferred_role = profile.role_title
            self.inferred_experience = self._infer_experience(resume_text)
            logger.info(
                "Built grounded ontology for %s with %d skills.",
                self.inferred_role,
                len(profile.skills),
            )
            return True
        except Exception as e:
            logger.error("Grounded ontology adaptation failed: %s", e)
            self.role_profile = None
            self.skill_metadata = {}
            return False

    @staticmethod
    def _infer_experience(resume_text: str) -> str:
        text = resume_text.casefold()
        years = [int(value) for value in re.findall(r"\b(\d{1,2})\+?\s+years?\b", text)]
        maximum = max(years, default=0)
        if maximum >= 7:
            return "Senior"
        if maximum >= 2:
            return "Mid-Level"
        return "Fresher"

    def _validate_graph(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("Grounded ontology contains a prerequisite cycle")
            if node in visited:
                return
            visiting.add(node)
            for successor in self.successors.get(node, ()):
                visit(successor)
            visiting.remove(node)
            visited.add(node)

        for node in self.nodes:
            visit(node)

    def get_skill_metadata(self, skill: str) -> GroundedSkill | None:
        return self.skill_metadata.get(skill)
        
    def get_prerequisites(self, skill):
        """
        Returns immediate prerequisite skills (incoming edges).
        Useful for 'probe_foundation' action.
        """
        if skill not in self.nodes:
            return []
        return list(self.predecessors.get(skill, []))
    
    def get_advanced(self, skill):
        """
        Returns immediate advanced skills (outgoing edges).
        Useful for 'increase_difficulty' action.
        """
        if skill not in self.nodes:
            return []
        return list(self.successors.get(skill, []))
        
    def get_all_skills(self):
        return list(self.nodes)

if __name__ == "__main__":
    # Test execution
    print("Testing Backend Developer Ontology...")
    ontology = SkillOntologyGraph("backend_developer")
    print(f"Nodes: {len(ontology.get_all_skills())}")
    print(f"Prerequisites for 'JWT': {ontology.get_prerequisites('JWT')}")
    print(f"Advanced from 'JWT': {ontology.get_advanced('JWT')}")
    
    print("\nTesting Dynamic Adaptation Fallback (No API Key)...")
    success = ontology.adapt_to_candidate("We need a MongoDB expert.", "I am a MongoDB expert.")
    print(f"Adaptation Success: {success}")
    print(f"Nodes after fallback: {len(ontology.get_all_skills())}")
