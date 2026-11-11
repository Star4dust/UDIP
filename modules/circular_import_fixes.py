"""
UDIP v3 — modules/circular_import_fixes.py
Fixes for circular import dependencies.

This module provides dependency injection and interfaces to break
circular imports while maintaining functionality.
"""

from typing import Protocol, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LoggerInterface(Protocol):
    """Interface for logging functionality."""
    
    def log(self, message: str, level: str = "info"):
        """Log a message with specified level."""
        ...


class StateManagerInterface(Protocol):
    """Interface for state management functionality."""
    
    def get_active_dataset(self) -> Optional[str]:
        """Get currently active dataset."""
        ...
    
    def set_active_dataset(self, dataset_id: str):
        """Set active dataset."""
        ...


class CoreManagerInterface(Protocol):
    """Interface for core functionality."""
    
    def add_history(self, page: str, action: str):
        """Add action to history."""
        ...
    
    def save_dataset(self, dataset_id: str, data: Dict[str, Any]) -> bool:
        """Save dataset with metadata."""
        ...


class DependencyInjector:
    """Manages dependency injection to break circular imports."""
    
    def __init__(self):
        self._services = {}
        self._factories = {}
    
    def register_service(self, name: str, service: Any):
        """Register a service instance."""
        self._services[name] = service
        logger.info(f"Registered service: {name}")
    
    def register_factory(self, name: str, factory: Callable[[], Any]):
        """Register a service factory."""
        self._factories[name] = factory
        logger.info(f"Registered factory: {name}")
    
    def get_service(self, name: str):
        """Get a service instance."""
        if name in self._services:
            return self._services[name]
        elif name in self._factories:
            service = self._factories[name]()
            self._services[name] = service
            return service
        else:
            logger.error(f"Service not found: {name}")
            return None
    
    def get_logger(self) -> LoggerInterface:
        """Get logger service."""
        return self.get_service("logger")
    
    def get_state_manager(self) -> StateManagerInterface:
        """Get state manager service."""
        return self.get_service("state_manager")
    
    def get_core_manager(self) -> CoreManagerInterface:
        """Get core manager service."""
        return self.get_service("core_manager")


# Global dependency injector
dependency_injector = DependencyInjector()


def get_dependency_injector() -> DependencyInjector:
    """Get global dependency injector."""
    return dependency_injector


class SafeLogger:
    """Safe logger implementation that doesn't create circular dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log(self, message: str, level: str = "info"):
        """Log message safely."""
        if level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)


class LazyImport:
    """Lazy import mechanism to avoid circular dependencies."""
    
    def __init__(self, module_path: str):
        self.module_path = module_path
        self._module = None
    
    def __getattr__(self, name: str):
        """Lazy load the module attribute."""
        if self._module is None:
            try:
                import importlib
                self._module = importlib.import_module(self.module_path)
                logger.info(f"Lazy loaded module: {self.module_path}")
            except ImportError as e:
                logger.error(f"Failed to lazy load {self.module_path}: {e}")
                raise AttributeError(f"Module {self.module_path} not available: {e}")
        
        return getattr(self._module, name)


class EventManager:
    """Event management system to reduce direct coupling."""
    
    def __init__(self):
        self._listeners = {}
    
    def subscribe(self, event_type: str, callback: Callable[[Any], None]):
        """Subscribe to an event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        logger.info(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]):
        """Unsubscribe from an event type."""
        if event_type in self._listeners:
            try:
                self._listeners[event_type].remove(callback)
                logger.info(f"Unsubscribed from event: {event_type}")
            except ValueError:
                logger.warning(f"Callback not found for event: {event_type}")
    
    def publish(self, event_type: str, data: Any):
        """Publish an event."""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
            logger.debug(f"Published event: {event_type}")
        else:
            logger.warning(f"No listeners for event: {event_type}")


# Global event manager
event_manager = EventManager()


def get_event_manager() -> EventManager:
    """Get global event manager."""
    return event_manager


class ConfigurationManager:
    """Configuration manager with dependency injection."""
    
    def __init__(self, injector: DependencyInjector):
        self.injector = injector
    
    def get_logger(self) -> LoggerInterface:
        """Get logger through dependency injection."""
        return self.injector.get_logger()
    
    def get_state_manager(self) -> StateManagerInterface:
        """Get state manager through dependency injection."""
        return self.injector.get_state_manager()
    
    def get_core_manager(self) -> CoreManagerInterface:
        """Get core manager through dependency injection."""
        return self.injector.get_core_manager()


# Register core services
dependency_injector.register_service("logger", SafeLogger())
dependency_injector.register_factory("state_manager", lambda: LazyImport("modules.state_manager").get_state_manager())
dependency_injector.register_factory("core_manager", lambda: LazyImport("modules.core"))


def get_configuration_manager() -> ConfigurationManager:
    """Get configuration manager with dependency injection."""
    return ConfigurationManager(dependency_injector)
