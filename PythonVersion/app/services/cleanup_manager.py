"""Cleanup manager - handles graceful shutdown of resources."""
import atexit
import asyncio
import signal
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.vllm_manager import VLLMServerManager


class CleanupManager:
    """Manages graceful shutdown of resources on application exit.
    
    Registers handlers for:
    - atexit (normal Python exit)
    - SIGINT (Ctrl+C)
    - SIGTERM (Docker/systemd termination)
    """
    
    _instance: Optional['CleanupManager'] = None
    
    def __init__(self, vllm_manager: Optional['VLLMServerManager'] = None):
        """
        Initialize the cleanup manager.
        
        Args:
            vllm_manager: Optional VLLMServerManager to cleanup on exit
        """
        self.vllm_manager = vllm_manager
        self._cleanup_done = False
        self._original_sigint = None
        self._original_sigterm = None
    
    @classmethod
    def get_instance(cls) -> 'CleanupManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = CleanupManager()
        return cls._instance
    
    def set_vllm_manager(self, vllm_manager: 'VLLMServerManager') -> None:
        """Set the vLLM manager to cleanup."""
        self.vllm_manager = vllm_manager
    
    def register_handlers(self) -> None:
        """Register all cleanup handlers."""
        # Register atexit handler
        atexit.register(self._cleanup_sync)
        
        # Register signal handlers
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("[CleanupManager] Handlers registered")
    
    def unregister_handlers(self) -> None:
        """Unregister signal handlers (restore originals)."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle SIGINT and SIGTERM signals."""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[CleanupManager] Received {sig_name}, cleaning up...")
        
        self._cleanup_sync()
        
        # Exit cleanly
        sys.exit(0)
    
    def _cleanup_sync(self) -> None:
        """Synchronous cleanup wrapper."""
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        
        # Try to run async cleanup
        try:
            # Check if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, schedule cleanup
                asyncio.create_task(self._cleanup_async())
            except RuntimeError:
                # No running loop, create one for cleanup
                asyncio.run(self._cleanup_async())
        except Exception as e:
            print(f"[CleanupManager] Error during cleanup: {e}")
            # Fallback: try direct process cleanup
            self._force_cleanup()
    
    async def _cleanup_async(self) -> None:
        """Async cleanup of all resources."""
        print("[CleanupManager] Running cleanup...")
        
        if self.vllm_manager and self.vllm_manager.is_running():
            print("[CleanupManager] Stopping vLLM server...")
            await self.vllm_manager.stop_server()
        
        print("[CleanupManager] Cleanup complete")
    
    def _force_cleanup(self) -> None:
        """Force cleanup without async (last resort)."""
        if self.vllm_manager and self.vllm_manager._process:
            try:
                self.vllm_manager._process.terminate()
                self.vllm_manager._process.wait(timeout=5)
            except Exception:
                try:
                    self.vllm_manager._process.kill()
                except Exception:
                    pass
