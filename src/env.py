import numpy as np
import pandas as pd
import joblib


class RansomwareEnv:
    def __init__(self, df_path, preproc_path="/content/ransomeware_detction/artifacts/preprocessor.pkl", target_col="Prediction"):
        
        # Load data + transformer
        self.df = pd.read_csv(df_path)
        self.preproc = joblib.load(preproc_path)

        # Process target
        self.target_col = target_col
        self.y = self.df[target_col].values

        # Remove target from state and transform
        X = self.df.drop(columns=[target_col])
        self.X = self.preproc.transform(X)  # final state vectors

        # Clusters define episodes
        self.cluster_ids = self.df["Clusters"].values
        self.clusters = np.unique(self.cluster_ids)

        # Episode state
        self.current_cluster = None
        self.current_idx = None
        self.active_indices = None

    def reset(self):
        """Start new episode by selecting next cluster group"""

        # Pick cluster (random or sequential)
        if not hasattr(self, "cluster_pointer"):
            self.cluster_pointer = 0

        if self.cluster_pointer >= len(self.clusters):
            self.cluster_pointer = 0

        self.current_cluster = self.clusters[self.cluster_pointer]
        self.cluster_pointer += 1

        # Slice rows of this cluster
        self.active_indices = np.where(self.cluster_ids == self.current_cluster)[0]
        self.pos = 0
        idx = self.active_indices[self.pos]

        return self.X[idx]

    def step(self, action):
        """
        RL step:
        action = {0,1,2}
        returns (next_state, reward, done, info)
        """

        idx = self.active_indices[self.pos]
        true_label = self.y[idx]

        reward = 1 if action == true_label else -1

        self.pos += 1
        done = self.pos >= len(self.active_indices)

        if not done:
            next_idx = self.active_indices[self.pos]
            next_state = self.X[next_idx]
        else:
            next_state = None

        return next_state, reward, done, {}

    def sample_state(self):
        """Convenience helper: return random state."""
        rand = np.random.randint(0, len(self.X))
        return self.X[rand]
