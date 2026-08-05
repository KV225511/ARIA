import json
import os
import logging
import re
import requests
from dotenv import load_dotenv

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
        logger.info("Starting dynamic ontology adaptation via local Ollama...")
        
        base_nodes = self.base_data.get("nodes", [])
        
        prompt = f"""
You are an expert technical interviewer and ontologist.
Below are the baseline skills for the role of {self.role_name}:
{json.dumps(base_nodes)}

Here is the Job Description:
{jd_text[:1500]}

Here is the Candidate's Resume:
{resume_text[:1500]}

Modify the baseline ontology graph to perfectly adapt to the candidate's resume and the job description.
- Add highly relevant new skills mentioned in the resume/JD as new nodes.
- Remove baseline skills that are completely irrelevant to the JD or resume.
- Update the edges to reflect prerequisite -> advanced relationships correctly.
ALSO, infer the specific Role Name and the Experience Level (Fresher, Mid-Level, or Senior).

Return ONLY valid JSON with this exact schema:
{{
    "inferred_role": "string",
    "inferred_experience": "string",
    "nodes": ["skill1", "skill2"], 
    "edges": [["prereq", "advanced"]]
}}
Do not include any markdown formatting. Output ONLY JSON.
"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 16384
            }
        }
        
        try:
            # Huge timeout for local generation (300s)
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            
            data = response.json()
            raw_text = data.get("response", "").strip()
                
            # Use Regex to extract just the JSON block in case the LLM was chatty
            json_match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)
            else:
                clean_json_str = raw_text
                
            parsed = json.loads(clean_json_str)
            
            self.inferred_role = parsed.get("inferred_role", self.role_name)
            self.inferred_experience = parsed.get("inferred_experience", "Mid-Level")
            
            if "nodes" not in parsed or "edges" not in parsed:
                raise ValueError("LLM returned JSON without 'nodes' or 'edges' keys.")
                
            self._clear_graph()
            for node in parsed.get("nodes", []):
                self._add_node(node)
            for edge in parsed.get("edges", []):
                if len(edge) == 2:
                    self._add_edge(edge[0], edge[1])
            
            logger.info(f"Successfully adapted ontology dynamically. Role: {self.inferred_role}, Exp: {self.inferred_experience}.")
            return True
                
        except Exception as e:
            logger.error(f"Dynamic adaptation failed via Ollama: {e}. Falling back to static graph.")
            try:
                self._load_graph() # Reset to baseline
            except Exception as load_e:
                logger.error(f"Fallback load also failed: {load_e}")
            self.inferred_role = self.role_name
            self.inferred_experience = "Mid-Level"
            return False
        
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
