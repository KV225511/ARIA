import math
import numpy as np

class BeliefStateUpdater:
    DEFAULT_BELIEF = np.array([0.333, 0.333, 0.334])

    def __init__(self, skill_nodes):
        """
        Initializes the belief state for all skills in the ontology.
        Each skill starts with a uniform distribution: [P(beginner), P(mid), P(expert)] = [0.33, 0.33, 0.33]
        """
        self.beliefs = {skill: self.DEFAULT_BELIEF.copy() for skill in skill_nodes}
        self.global_entropy_sum = sum(self._calculate_entropy(dist) for dist in self.beliefs.values())
        
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
        if not self.beliefs:
            return 0.0
        return self.global_entropy_sum / len(self.beliefs)
        
    def get_belief(self, skill):
        return self.beliefs.get(skill, self.DEFAULT_BELIEF.copy())
        
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
            return self.DEFAULT_BELIEF.copy()
            
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
        # Smooth Gaussian likelihoods (variance = 0.25) to prevent extreme single-turn collapse
        # Beginner centered around 0.2, Mid around 0.5, Expert around 0.8
        
        beginner = math.exp(-((evidence_score - 0.2) ** 2) / 0.25)
        mid = math.exp(-((evidence_score - 0.5) ** 2) / 0.25)
        expert = math.exp(-((evidence_score - 0.8) ** 2) / 0.25)
        
        likelihood = np.array([beginner, mid, expert])
        likelihood = self._normalize(likelihood)
            
        # Bayesian update: Posterior propto Prior * Likelihood
        unnormalized_posterior = current_belief * likelihood
        new_belief = self._normalize(unnormalized_posterior)
        
        # Update running entropy sum
        old_entropy = self._calculate_entropy(current_belief)
        new_entropy = self._calculate_entropy(new_belief)
        self.global_entropy_sum += (new_entropy - old_entropy)
        
        self.beliefs[skill] = new_belief
        
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
