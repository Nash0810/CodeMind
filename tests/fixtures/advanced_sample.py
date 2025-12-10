"""Advanced test fixture with comprehensive metadata."""

from typing import List, Optional
from abc import ABC, abstractmethod


@staticmethod
def simple_utility():
    """A simple utility function."""
    return 42


@staticmethod
def cached_function(value: int) -> int:
    """A function with caching decorator."""
    return value * 2


def greet(name: str, greeting: str = "Hello") -> str:
    """
    Greets a person with a custom greeting.
    
    Args:
        name: The person's name
        greeting: Custom greeting message
        
    Returns:
        Formatted greeting string
    """
    return f"{greeting}, {name}!"


async def fetch_data(url: str) -> Optional[dict]:
    """
    Asynchronously fetches data from URL.
    """
    # This is a mock implementation
    return {"data": "from_url"}


class Animal(ABC):
    """
    Abstract base class for animals.
    """
    
    def __init__(self, name: str, age: int):
        """Initialize an animal."""
        self.name = name
        self.age = age
    
    @abstractmethod
    def speak(self) -> str:
        """Make the animal speak - must be implemented by subclasses."""
        pass
    
    def get_info(self) -> str:
        """Get animal information."""
        info = self.speak()
        return info


class Dog(Animal):
    """
    A dog class that extends Animal.
    """
    
    def __init__(self, name: str, age: int, breed: str = "Mixed"):
        """Initialize a dog with breed."""
        super().__init__(name, age)
        self.breed = breed
    
    def speak(self) -> str:
        """Dogs bark."""
        return "Woof!"
    
    def fetch(self, item: str) -> bool:
        """Fetch an item."""
        return True
    
    @staticmethod
    def create_puppy(name: str) -> "Dog":
        """Create a new puppy."""
        return Dog(name, 0, "Puppy")


class DataProcessor:
    """Processes data with various transformations."""
    
    def __init__(self, data: List[int]):
        """Initialize with data."""
        self.data = data
    
    def filter_positive(self) -> List[int]:
        """Filter positive numbers."""
        return [x for x in self.data if x > 0]
    
    def transform(self, multiplier: int = 2) -> List[int]:
        """Transform data by multiplication."""
        result = [x * multiplier for x in self.data]
        return result
