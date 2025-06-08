# core/hierarchical_memory.py

class ShortTermMemory:
    """Represents the short-term memory component."""
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.memory = []

    def add(self, item):
        """Adds an item to short-term memory, managing capacity."""
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            # Simple eviction strategy: remove the oldest item
            self.memory.pop(0)

    def retrieve(self, query):
        """Retrieves items from short-term memory based on a query."""
        # Simple retrieval: return all items for now
        return self.memory

    def clear(self):
        """Clears the short-term memory."""
        self.memory = []

class MediumTermMemory:
    """Represents the medium-term memory component."""
    def __init__(self):
        self.memory = {} # Using a dictionary for potential key-based access

    def add(self, key, item):
        """Adds an item to medium-term memory with a key."""
        self.memory[key] = item

    def retrieve(self, key):
        """Retrieves an item from medium-term memory by key."""
        return self.memory.get(key)

    def clear(self):
        """Clears the medium-term memory."""
        self.memory = {}
 
class ThinkingMemory:
    """Represents the temporary memory component for the current reasoning cycle."""
    def __init__(self):
        self.memory = [] # Using a list to store items relevant to the current thought process
 
    def add(self, item):
        """Adds an item to thinking memory."""
        self.memory.append(item)
 
    def retrieve(self, query=None):
        """Retrieves all items from thinking memory. Query is ignored for now."""
        return self.memory
 
    def clear(self):
        """Clears the thinking memory."""
        self.memory = []
 
class LongTermMemory:
    """Represents the long-term memory component."""
    def __init__(self):
        # Placeholder for long-term storage mechanism (e.g., file, database)
        self.memory = [] # Using a list as a simple placeholder
 
    def add(self, item):
        """Adds an item to long-term memory."""
        self.memory.append(item) # Placeholder
 
    def retrieve(self, query):
        """Retrieves items from long-term memory based on a query."""
        # Simple retrieval: return all items for now
        return self.memory # Placeholder
 
    def clear(self):
        """Clears the long-term memory."""
        self.memory = [] # Placeholder