class Environment:
    def __init__(self):
        pass
    
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        raise NotImplementedError
    
    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, and done flag.
        """
        raise NotImplementedError
