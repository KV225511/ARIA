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
Below is the baseline skill ontology graph for the role of {self.role_name} in JSON format:
{json.dumps(self.base_data)}

Here is the Job Description:
{job_description}

Here is the Candidate's Resume:
{resume}

Modify the baseline ontology graph to perfectly adapt to the candidate's resume and the job description.
- Add highly relevant new skills mentioned in the resume/JD as new nodes.
- Remove baseline skills that are completely irrelevant to the JD or resume.
- Update the edges to reflect prerequisite -> advanced relationships correctly.
Return ONLY valid JSON with the format: {{"nodes": [...], "edges": [["prereq", "advanced"], ...]}}
Do not include any markdown formatting (like ```json), just the raw JSON object.
"""
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            }
            
            response = requests.post(api_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            response_text = response.json().get("response", "").strip()
            
            # Clean up potential markdown formatting from the response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            adapted_data = json.loads(response_text)
            
            # Save the adapted graph for inspection
            debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_adapted_ontology.json")
            with open(debug_path, "w") as f:
                json.dump(adapted_data, f, indent=4)
            
            # Validate structure before applying
            if "nodes" not in adapted_data or "edges" not in adapted_data:
                raise ValueError("LLM returned JSON without 'nodes' or 'edges' keys.")
                
            self.graph.clear()
            self.graph.add_nodes_from(adapted_data.get("nodes", []))
            self.graph.add_edges_from(adapted_data.get("edges", []))
            logger.info(f"Successfully adapted ontology graph dynamically via {model}.")
            return True
            
        except Exception as e:
            logger.error(f"Dynamic adaptation failed via Ollama: {e}. Falling back to static graph.")
            self._load_graph() # Reset to baseline
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
