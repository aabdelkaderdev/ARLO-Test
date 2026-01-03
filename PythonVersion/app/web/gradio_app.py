"""Gradio web interface for ARLO."""
import gradio as gr
import pandas as pd
from typing import Tuple, Optional, List
import asyncio
import os
import tempfile
from datetime import datetime

from app.architect import Architect, QualityWeightsMode as ArchitectQualityWeightsMode
from app.services.ollama_service import OllamaService
from app.services.vllm_service import VLLMService
from app.services.vllm_manager import VLLMServerManager
from app.services.optimizer_service import OptimizerMode
from app.web.pdf_generator import generate_pdf_report


# Available Ollama models
AVAILABLE_OLLAMA_MODELS = [
    "llama3.1:latest",
    "llama3.1:70b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "deepseek-coder-v2:latest",
    "deepseek-coder:latest",
    "qwen2.5-coder:32b",
    "gemma2:latest",
]

# Global vLLM manager instance
vllm_manager: Optional[VLLMServerManager] = None


def get_vllm_manager() -> VLLMServerManager:
    """Get or create the global vLLM manager instance."""
    global vllm_manager
    if vllm_manager is None:
        vllm_manager = VLLMServerManager(log_callback=None)
    return vllm_manager


def parse_txt_file(file_path: str) -> list[str]:
    """Parse a .txt file and extract requirements (one per line)."""
    if file_path is None:
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Filter empty lines and strip whitespace
    requirements = [line.strip() for line in lines if line.strip()]
    return requirements


def get_available_models(backend: str) -> List[str]:
    """Get available models based on backend selection."""
    if backend == "vLLM":
        manager = get_vllm_manager()
        models = manager.get_available_models()
        if not models:
            return ["(No models found in ~/vllm_models/hub/)"]
        return models
    else:
        return AVAILABLE_OLLAMA_MODELS


def get_gpu_choices() -> List[int]:
    """Get available GPU choices."""
    manager = get_vllm_manager()
    num_gpus = manager.get_available_gpus()
    if num_gpus == 0:
        return [0]  # No GPUs available
    return list(range(1, min(num_gpus + 1, 3)))  # 1 or 2 GPUs max


def update_model_dropdown(backend: str):
    """Update model dropdown based on backend selection."""
    models = get_available_models(backend)
    return gr.update(choices=models, value=models[0] if models else None)


def update_vllm_visibility(backend: str):
    """Update visibility of vLLM-specific controls."""
    is_vllm = backend == "vLLM"
    return (
        gr.update(visible=is_vllm),  # gpu_selector
        gr.update(visible=is_vllm),  # advanced_options
        gr.update(visible=is_vllm),  # logs_section
    )


def format_time_remaining(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds <= 0:
        return "Complete"
    elif seconds < 60:
        return f"{int(seconds)} sec"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins} min {secs} sec"


async def run_analysis(
    requirements: list[str],
    optimization_strategy: str,
    quality_weights_mode: str,
    backend: str,
    model: str,
    num_gpus: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict:
    """Run the ARLO analysis on requirements."""
    opt_mode = OptimizerMode.ILP if optimization_strategy == "ILP" else OptimizerMode.GREEDY
    
    weights_mode_map = {
        "Equally Important": ArchitectQualityWeightsMode.EQUALLY_IMPORTANT,
        "Inferred": ArchitectQualityWeightsMode.INFERRED,
    }
    weights_mode = weights_mode_map.get(quality_weights_mode, ArchitectQualityWeightsMode.INFERRED)
    
    llm_service = None
    
    try:
        if backend == "vLLM":
            # Start vLLM server if needed
            manager = get_vllm_manager()
            manager.clear_logs()
            
            # Check if model is valid
            if model.startswith("(No models"):
                return {
                    "success": False,
                    "error": "No vLLM models found. Please add models to ~/vllm_models/hub/",
                    "concerns": [],
                    "asrs": [],
                    "requirements": [],
                    "report": "",
                }
            
            # Start server
            started = await manager.start_server(
                model_path=model,
                num_gpus=num_gpus,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            
            if not started:
                return {
                    "success": False,
                    "error": "Failed to start vLLM server. Check logs for details.",
                    "concerns": [],
                    "asrs": [],
                    "requirements": [],
                    "report": "",
                }
            
            # Wait for server to be ready
            ready = await manager.wait_for_ready(timeout=300)
            
            if not ready:
                return {
                    "success": False,
                    "error": "vLLM server failed to become ready. Check logs for details.",
                    "concerns": [],
                    "asrs": [],
                    "requirements": [],
                    "report": "",
                }
            
            # Create vLLM service
            llm_service = VLLMService(model=model)
        else:
            # Use Ollama
            llm_service = OllamaService(model=model)
        
        architect = Architect(ollama_service=llm_service)
        requirements_text = "\n".join(requirements)
        
        concerns, report = await architect.analyze(
            requirements_text=requirements_text,
            optimization_mode=opt_mode,
            quality_weights_mode=weights_mode,
        )
        
        # Start idle timer for vLLM
        if backend == "vLLM":
            manager = get_vllm_manager()
            manager.start_idle_timer()
        
        return {
            "success": True,
            "concerns": concerns,
            "asrs": architect.asrs,
            "requirements": architect.requirements,
            "report": report,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "concerns": [],
            "asrs": [],
            "requirements": [],
            "report": "",
        }
    finally:
        if llm_service:
            await llm_service.close()


def create_asrs_dataframe(asrs: list) -> pd.DataFrame:
    """Create a DataFrame for ASRs table."""
    if not asrs:
        return pd.DataFrame(columns=["ID", "Description", "Quality Attributes", "Condition"])
    
    data = []
    for asr in asrs:
        data.append({
            "ID": asr.id,
            "Description": asr.description[:100] + "..." if len(asr.description) > 100 else asr.description,
            "Quality Attributes": ", ".join(asr.quality_attributes) if asr.quality_attributes else "-",
            "Condition": asr.condition_text or "-",
        })
    
    return pd.DataFrame(data)


def create_decisions_dataframe(concerns: list) -> pd.DataFrame:
    """Create a DataFrame for all decisions across concerns."""
    if not concerns:
        return pd.DataFrame(columns=["Concern", "Pattern Type", "Selected Pattern", "Score", "Satisfied", "Tradeoffs"])
    
    data = []
    for i, concern in enumerate(concerns, 1):
        condition_label = ", ".join(concern.conditions[:2]) if concern.conditions else f"Concern {i}"
        if len(condition_label) > 40:
            condition_label = condition_label[:37] + "..."
        
        for decision in concern.decisions:
            satisfied = ", ".join([f"{q}({s})" for q, s in decision.satisfied_qualities]) if decision.satisfied_qualities else "-"
            tradeoffs = ", ".join([f"{q}({s})" for q, s in decision.unsatisfied_qualities]) if decision.unsatisfied_qualities else "-"
            
            data.append({
                "Concern": condition_label,
                "Pattern Type": decision.arch_pattern_name,
                "Selected Pattern": decision.selected_pattern,
                "Score": decision.score,
                "Satisfied": satisfied,
                "Tradeoffs": tradeoffs,
            })
    
    return pd.DataFrame(data)


def analyze_requirements(
    file,
    optimization_strategy: str,
    quality_weights_mode: str,
    backend: str,
    model: str,
    num_gpus: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, str, Optional[str], str]:
    """Main analysis function for Gradio interface."""
    if file is None:
        return (
            "⚠️ Please upload a .txt file with requirements.",
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            None,
            "",
        )
    
    # Parse requirements from file
    requirements = parse_txt_file(file)
    
    if not requirements:
        return (
            "⚠️ No requirements found in the file. Please ensure each requirement is on a separate line.",
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            None,
            "",
        )
    
    # Run analysis
    result = asyncio.run(run_analysis(
        requirements, 
        optimization_strategy, 
        quality_weights_mode,
        backend,
        model,
        num_gpus,
        max_model_len,
        gpu_memory_utilization,
    ))
    
    # Get logs if vLLM
    logs = ""
    if backend == "vLLM":
        manager = get_vllm_manager()
        logs = manager.get_logs(max_lines=1000)
    
    if not result["success"]:
        return (
            f"❌ Analysis failed: {result.get('error', 'Unknown error')}",
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            None,
            logs,
        )
    
    # Create summary
    total_reqs = len(result["requirements"])
    asr_count = len(result["asrs"])
    concern_count = len(result["concerns"])
    
    summary = f"""## ✅ Analysis Complete

| Metric | Value |
|--------|-------|
| Total Requirements | {total_reqs} |
| Architecturally-Significant | {asr_count} |
| Condition Groups | {concern_count} |
"""
    
    # Create dataframes
    asrs_df = create_asrs_dataframe(result["asrs"])
    decisions_df = create_decisions_dataframe(result["concerns"])
    
    # Generate PDF
    pdf_path = generate_pdf_report(result)
    
    return (
        summary,
        asrs_df,
        decisions_df,
        result["report"],
        pdf_path,
        logs,
    )


def clear_vllm_logs():
    """Clear the vLLM logs display."""
    manager = get_vllm_manager()
    manager.clear_logs()
    return ""


def export_vllm_logs():
    """Export all vLLM logs to a file."""
    manager = get_vllm_manager()
    all_logs = manager.get_all_logs()
    
    if not all_logs:
        return None
    
    # Create temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vllm_logs_{timestamp}.log"
    
    # Write to temp file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(all_logs)
    
    return file_path


def refresh_logs():
    """Refresh the logs display."""
    manager = get_vllm_manager()
    return manager.get_logs(max_lines=1000)


def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    
    with gr.Blocks(
        title="ARLO - Architectural Requirements Optimizer",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
# 🏗️ ARLO - Architectural Requirements to Logical Optimization

Upload your software requirements file (.txt) and get optimized architectural decisions.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Requirements (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                
                # Backend selection
                backend_selector = gr.Radio(
                    choices=["Ollama", "vLLM"],
                    value="Ollama",
                    label="LLM Backend",
                )
                
                optimization_strategy = gr.Radio(
                    choices=["ILP", "Greedy"],
                    value="ILP",
                    label="Optimization Strategy",
                )
                
                quality_weights_mode = gr.Radio(
                    choices=["Inferred", "Equally Important"],
                    value="Inferred",
                    label="Quality Weights Mode",
                )
                
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_OLLAMA_MODELS,
                    value="llama3.1:latest",
                    label="Model",
                )
                
                # GPU selector (vLLM only)
                gpu_choices = get_gpu_choices()
                gpu_selector = gr.Dropdown(
                    choices=gpu_choices,
                    value=gpu_choices[0] if gpu_choices else 1,
                    label="Number of GPUs",
                    visible=False,
                )
                
                # Advanced vLLM options
                with gr.Accordion("Advanced vLLM Options", open=False, visible=False) as advanced_options:
                    max_model_len = gr.Number(
                        value=4096,
                        label="Max Model Length",
                        precision=0,
                    )
                    gpu_memory_utilization = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.80,
                        step=0.05,
                        label="GPU Memory Utilization",
                    )
                
                # Time estimation
                time_remaining = gr.Textbox(
                    label="Estimated Time Remaining",
                    value="--:--",
                    interactive=False,
                )
                
                analyze_btn = gr.Button("🔍 Analyze Requirements", variant="primary")
        
        with gr.Row():
            summary_output = gr.Markdown(label="Summary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Architecturally-Significant Requirements (ASRs)")
                asrs_table = gr.Dataframe(
                    headers=["ID", "Description", "Quality Attributes", "Condition"],
                    interactive=False,
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Architectural Decisions")
                decisions_table = gr.Dataframe(
                    headers=["Concern", "Pattern Type", "Selected Pattern", "Score", "Satisfied", "Tradeoffs"],
                    interactive=False,
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Full Report")
                report_output = gr.Textbox(
                    label="Text Report",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                )
        
        with gr.Row():
            pdf_output = gr.File(label="📥 Download PDF Report")
        
        # vLLM Logs Section
        with gr.Row(visible=False) as logs_section:
            with gr.Column():
                gr.Markdown("### 🔧 vLLM Server Logs")
                vllm_logs = gr.Textbox(
                    label="Server Logs",
                    lines=15,
                    max_lines=25,
                    interactive=False,
                    autoscroll=True,
                )
                with gr.Row():
                    clear_logs_btn = gr.Button("Clear Logs", size="sm")
                    export_logs_btn = gr.Button("Export Logs", size="sm")
                    refresh_logs_btn = gr.Button("Refresh Logs", size="sm")
                log_file_output = gr.File(label="Download Logs", visible=False)
        
        # Event handlers
        backend_selector.change(
            fn=update_model_dropdown,
            inputs=[backend_selector],
            outputs=[model_dropdown],
        )
        
        backend_selector.change(
            fn=update_vllm_visibility,
            inputs=[backend_selector],
            outputs=[gpu_selector, advanced_options, logs_section],
        )
        
        analyze_btn.click(
            fn=analyze_requirements,
            inputs=[
                file_input, 
                optimization_strategy, 
                quality_weights_mode, 
                backend_selector,
                model_dropdown,
                gpu_selector,
                max_model_len,
                gpu_memory_utilization,
            ],
            outputs=[
                summary_output, 
                asrs_table, 
                decisions_table, 
                report_output, 
                pdf_output,
                vllm_logs,
            ],
        )
        
        clear_logs_btn.click(
            fn=clear_vllm_logs,
            outputs=[vllm_logs],
        )
        
        export_logs_btn.click(
            fn=export_vllm_logs,
            outputs=[log_file_output],
        )
        
        refresh_logs_btn.click(
            fn=refresh_logs,
            outputs=[vllm_logs],
        )
    
    return app


# Create the app instance
gradio_app = create_gradio_app()
