import math
import numpy as np

class BeliefStateUpdater:
    def __init__(self, skill_nodes):
        """
        Initializes the belief state for all skills in the ontology.
        Each skill starts with a uniform distribution: [P(beginner), P(mid), P(expert)] = [0.33, 0.33, 0.33]
        """
        self.beliefs = {skill: np.array([0.333, 0.333, 0.334]) for skill in skill_nodes}
        
    def _normalize(self, dist):
        total = np.sum(dist)
        if total == 0:
            return np.ones_like(dist) / len(dist)
        return dist / total
        
    def _calculate_entropy(self, dist):
        # Shannon entropy for a distribution
        return -np.sum(dist * np.log(dist + 1e-9))
        
    def get_global_entropy(self):
        """
        Returns average entropy across all node beliefs.
        Used as the termination signal.
        """
        entropies = [self._calculate_entropy(dist) for dist in self.beliefs.values()]
        if not entropies:
            return 0.0
        return np.mean(entropies)
        
    def get_belief(self, skill):
        return self.beliefs.get(skill, np.array([0.333, 0.333, 0.334]))
        
    def update_belief(self, skill, semantic_score, cognitive_load, behavior_score):
        """
        Bayesian update for a specific skill node.
        
        Args:
            skill: string name of the node
            semantic_score: 0.0 to 1.0 (0=shallow, 1=strong)
            cognitive_load: 'low', 'anxiety', or 'ignorance'
            behavior_score: 0.0 to 1.0 (confidence, gaze, prosody)
        """
        if skill not in self.beliefs:
            return
            
        current_belief = self.beliefs[skill]
        
        # Determine likelihoods based on semantic and behavioral scores
        # L = [L(beginner), L(mid), L(expert)]
        
        # Adjust weight based on cognitive load
        # If anxious, rely heavily on semantic content, disregard bad behavior
        if cognitive_load == 'anxiety':
            weight_semantic = 0.9
            weight_behavior = 0.1
        elif cognitive_load == 'ignorance':
            weight_semantic = 0.5
            weight_behavior = 0.5
        else: # low load
            weight_semantic = 0.6
            weight_behavior = 0.4
            
        # Combine evidence
        evidence_score = (semantic_score * weight_semantic) + (behavior_score * weight_behavior)
        
        # Convert evidence into likelihood distribution for [beginner, mid, expert]
        # High evidence points to expert, low evidence points to beginner
        if evidence_score > 0.7:
            likelihood = np.array([0.1, 0.3, 0.6])
        elif evidence_score > 0.4:
            likelihood = np.array([0.2, 0.6, 0.2])
        else:
            likelihood = np.array([0.6, 0.3, 0.1])
            
        # Bayesian update: Posterior propto Prior * Likelihood
        unnormalized_posterior = current_belief * likelihood
        self.beliefs[skill] = self._normalize(unnormalized_posterior)
        
        return self.beliefs[skill]

if __name__ == "__main__":
    # Test execution
    nodes = ["REST API", "SQL"]
    updater = BeliefStateUpdater(nodes)
    
    print(f"Initial Global Entropy: {updater.get_global_entropy():.4f}")
    
    # Simulate a strong, confident turn for REST API
    print("\nUpdating REST API (Strong Semantic, High Confidence, Low Load)...")
    updater.update_belief("REST API", semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
    print(f"New Belief: {updater.get_belief('REST API')}")
    print(f"Global Entropy: {updater.get_global_entropy():.4f}")
    
    # Simulate an anxious but semantically correct turn for SQL
    print("\nUpdating SQL (Strong Semantic, Poor Behavior due to Anxiety)...")
    updater.update_belief("SQL", semantic_score=0.8, cognitive_load="anxiety", behavior_score=0.2)
    print(f"New Belief: {updater.get_belief('SQL')}")
    print(f"Global Entropy: {updater.get_global_entropy():.4f}")
