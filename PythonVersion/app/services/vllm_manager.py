"""vLLM server manager - handles vLLM server lifecycle."""
import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime


class VLLMServerManager:
    """Manages vLLM server lifecycle including start, stop, and health monitoring.
    
    Features:
    - Auto-discover models in ~/vllm_models/hub/
    - Start/stop vLLM server with configurable GPU settings
    - Log capture for UI display
    - Idle timeout for auto-shutdown
    - GPU detection
    """
    
    VLLM_PORT = 4568
    VLLM_MODELS_PATH = os.path.expanduser("~/vllm_models/hub")
    IDLE_TIMEOUT_SECONDS = 30
    
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the vLLM server manager.
        
        Args:
            log_callback: Optional callback to receive log lines for UI display
        """
        self._process: Optional[subprocess.Popen] = None
        self._current_model: Optional[str] = None
        self._log_callback = log_callback
        self._logs: List[str] = []
        self._max_log_lines = 10000  # Store all logs for export
        self._log_task: Optional[asyncio.Task] = None
        self._idle_timer_task: Optional[asyncio.Task] = None
    
    def get_available_models(self) -> List[str]:
        """
        Auto-discover models in ~/vllm_models/hub/.
        
        Returns:
            List of model folder names (e.g., 'models--meta-llama--Llama-3.1-8B')
        """
        models_path = Path(self.VLLM_MODELS_PATH)
        
        if not models_path.exists():
            return []
        
        models = []
        for item in models_path.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                models.append(item.name)
        
        return sorted(models)
    
    def get_available_gpus(self) -> int:
        """
        Detect the number of available GPUs.
        
        Returns:
            Number of available GPUs (0 if none or detection fails)
        """
        # Try torch first
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            pass
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return len(lines)
        except Exception:
            pass
        
        return 0
    
    async def start_server(
        self,
        model_path: str,
        num_gpus: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.80,
    ) -> bool:
        """
        Start the vLLM server with the specified configuration.
        
        Args:
            model_path: Model folder name (e.g., 'models--meta-llama--Llama-3.1-8B')
            num_gpus: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            
        Returns:
            True if server started successfully, False otherwise
        """
        # Stop existing server if running with different model
        if self._process and self._current_model != model_path:
            await self.stop_server()
        
        # If already running with same model, just reset idle timer
        if self._process and self._current_model == model_path:
            self._cancel_idle_timer()
            return True
        
        # Build full model path
        full_model_path = os.path.join(self.VLLM_MODELS_PATH, model_path)
        
        if not os.path.exists(full_model_path):
            self._log(f"ERROR: Model path does not exist: {full_model_path}")
            return False
        
        # Build command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", full_model_path,
            "--port", str(self.VLLM_PORT),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--enforce-eager",
        ]
        
        # Add tensor parallelism if using multiple GPUs
        if num_gpus > 1:
            cmd.extend(["--tensor-parallel-size", str(num_gpus)])
        
        self._log(f"Starting vLLM server: {' '.join(cmd)}")
        
        try:
            # Check if port is in use
            if self._is_port_in_use():
                self._log(f"ERROR: Port {self.VLLM_PORT} is already in use")
                return False
            
            # Start the process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            self._current_model = model_path
            
            # Start log capture task
            self._log_task = asyncio.create_task(self._capture_logs())
            
            self._log(f"vLLM server process started (PID: {self._process.pid})")
            return True
            
        except FileNotFoundError:
            self._log("ERROR: vLLM is not installed. Install with: pip install vllm")
            return False
        except Exception as e:
            self._log(f"ERROR: Failed to start vLLM server: {e}")
            return False
    
    async def stop_server(self) -> bool:
        """
        Stop the vLLM server gracefully.
        
        Returns:
            True if server was stopped, False if no server was running
        """
        self._cancel_idle_timer()
        
        if self._process is None:
            return False
        
        self._log("Stopping vLLM server...")
        
        try:
            # Send SIGTERM for graceful shutdown
            self._process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                self._process.wait(timeout=10)
                self._log("vLLM server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self._log("Graceful shutdown timed out, forcing termination...")
                self._process.kill()
                self._process.wait(timeout=5)
                self._log("vLLM server killed")
            
        except Exception as e:
            self._log(f"Error stopping vLLM server: {e}")
        finally:
            self._process = None
            self._current_model = None
            
            # Cancel log capture task
            if self._log_task:
                self._log_task.cancel()
                try:
                    await self._log_task
                except asyncio.CancelledError:
                    pass
                self._log_task = None
        
        return True
    
    def is_running(self) -> bool:
        """Check if the vLLM server is running."""
        if self._process is None:
            return False
        
        return self._process.poll() is None
    
    async def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the vLLM server to be ready (responding to health checks).
        
        Args:
            timeout: Optional timeout in seconds (None = no timeout)
            
        Returns:
            True if server is ready, False if timed out or failed
        """
        import httpx
        
        url = f"http://localhost:{self.VLLM_PORT}/v1/models"
        start_time = asyncio.get_event_loop().time()
        
        self._log("Waiting for vLLM server to be ready...")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                # Check if process is still running
                if not self.is_running():
                    self._log("ERROR: vLLM server process terminated unexpectedly")
                    return False
                
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        self._log("vLLM server is ready!")
                        return True
                except Exception:
                    pass
                
                # Check timeout
                if timeout is not None:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= timeout:
                        self._log(f"Timeout waiting for vLLM server after {timeout}s")
                        return False
                
                await asyncio.sleep(1)
    
    def start_idle_timer(self) -> None:
        """Start the idle timer for auto-shutdown after analysis completes."""
        self._cancel_idle_timer()
        self._idle_timer_task = asyncio.create_task(self._idle_timer())
    
    def _cancel_idle_timer(self) -> None:
        """Cancel the idle timer if running."""
        if self._idle_timer_task:
            self._idle_timer_task.cancel()
            self._idle_timer_task = None
    
    async def _idle_timer(self) -> None:
        """Timer that stops the server after idle timeout."""
        try:
            self._log(f"Idle timer started ({self.IDLE_TIMEOUT_SECONDS}s)")
            await asyncio.sleep(self.IDLE_TIMEOUT_SECONDS)
            self._log("Idle timeout reached, stopping server...")
            await self.stop_server()
        except asyncio.CancelledError:
            self._log("Idle timer cancelled")
    
    async def _capture_logs(self) -> None:
        """Capture stdout/stderr from the vLLM process."""
        if self._process is None or self._process.stdout is None:
            return
        
        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.readline
                )
                
                if not line and self._process.poll() is not None:
                    break
                
                if line:
                    self._log(line.rstrip())
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._log(f"Log capture error: {e}")
    
    def _log(self, message: str) -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        self._logs.append(log_line)
        
        # Trim logs if exceeding max
        if len(self._logs) > self._max_log_lines:
            self._logs = self._logs[-self._max_log_lines:]
        
        # Call UI callback
        if self._log_callback:
            self._log_callback(log_line)
        
        # Also print to console
        print(f"[vLLM] {message}")
    
    def _is_port_in_use(self) -> bool:
        """Check if the vLLM port is already in use."""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.VLLM_PORT)) == 0
    
    def get_logs(self, max_lines: int = 1000) -> str:
        """Get recent logs as a string."""
        recent_logs = self._logs[-max_lines:]
        return "\n".join(recent_logs)
    
    def get_all_logs(self) -> str:
        """Get all captured logs for export."""
        return "\n".join(self._logs)
    
    def clear_logs(self) -> None:
        """Clear the log buffer."""
        self._logs.clear()
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model name."""
        return self._current_model
