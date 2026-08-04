import json
import os
import networkx as nx
import logging

# Configure logger
logger = logging.getLogger(__name__)

class SkillOntologyGraph:
    def __init__(self, role_name="backend_developer"):
        """
        Initializes the skill ontology graph by loading the given role JSON.
        """
        self.role_name = role_name
        self.graph = nx.DiGraph()
        self._load_graph()
        
    def _load_graph(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "roles", f"{self.role_name}.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Role file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.graph.clear()
        self.graph.add_nodes_from(data.get("nodes", []))
        self.graph.add_edges_from(data.get("edges", []))
        self.base_data = data # Store base data for dynamic adaptation
        
    def adapt_to_candidate(self, job_description: str, resume: str) -> bool:
        """
        Dynamically adapts the ontology graph to the specific candidate and JD
        using a local Ollama model. If it fails, falls back to the static base graph.
        
        Returns:
            bool: True if dynamic adaptation succeeded, False if it fell back to static.
        """
        try:
            import requests
            from dotenv import load_dotenv
            load_dotenv()
            
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3.1")
            api_endpoint = f"{ollama_host}/api/generate"
            
            prompt = f"""
You are an expert technical interviewer and ontologist.
    def adapt_to_candidate(self, jd_text: str, resume_text: str) -> bool:
        """
        Takes JD and Resume text, uses local LLM to dynamically add missing skills.
        Also infers the target role and experience level.
        Returns True if successful, False if fell back to baseline.
        """
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.api_endpoint = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate"
        
        logger.info("Starting dynamic ontology adaptation via local Ollama...")
        
        prompt = f"""
You are an expert technical interviewer and ontologist.
Below is the baseline skill ontology graph for the role of {self.role_name} in JSON format:
{json.dumps(self.base_data)}

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
            
            # Save raw output for debugging
            debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_ollama_output.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
                
            # Use Regex to extract just the JSON block in case the LLM was chatty
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)
            else:
                clean_json_str = raw_text
                
            parsed = json.loads(clean_json_str)
            
            self.inferred_role = parsed.get("inferred_role", self.role_name)
            self.inferred_experience = parsed.get("inferred_experience", "Mid-Level")
            
            if "nodes" not in parsed or "edges" not in parsed:
                raise ValueError("LLM returned JSON without 'nodes' or 'edges' keys.")
                
            self.graph.clear()
            self.graph.add_nodes_from(parsed.get("nodes", []))
            self.graph.add_edges_from(parsed.get("edges", []))
            
            # Save the adapted ontology for debugging
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_adapted_ontology.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=4)
                
            logger.info(f"Successfully adapted ontology dynamically. Role: {self.inferred_role}, Exp: {self.inferred_experience}.")
            return True
                
        except Exception as e:
            logger.error(f"Dynamic adaptation failed via Ollama: {e}. Falling back to static graph.")
            self._load_graph() # Reset to baseline
            self.inferred_role = self.role_name
            self.inferred_experience = "Mid-Level"
            return False
        
    def get_prerequisites(self, skill):
        """
        Returns immediate prerequisite skills (incoming edges).
        Useful for 'probe_foundation' action.
        """
        if skill not in self.graph:
            return []
        return list(self.graph.predecessors(skill))
    
    def get_advanced(self, skill):
        """
        Returns immediate advanced skills (outgoing edges).
        Useful for 'increase_difficulty' action.
        """
        if skill not in self.graph:
            return []
        return list(self.graph.successors(skill))
        
    def get_all_skills(self):
        return list(self.graph.nodes)

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
