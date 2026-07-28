import json
import os
import networkx as nx

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
            
        self.graph.add_nodes_from(data.get("nodes", []))
        self.graph.add_edges_from(data.get("edges", []))
        
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
