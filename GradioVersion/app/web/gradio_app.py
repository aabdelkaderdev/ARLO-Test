"""Gradio web interface for ARLO."""
import gradio as gr
import pandas as pd
from typing import Tuple, Optional
import asyncio
import os

from app.architect import Architect, QualityWeightsMode as ArchitectQualityWeightsMode
from app.services.ollama_service import OllamaService
from app.services.optimizer_service import OptimizerMode
from app.web.pdf_generator import generate_pdf_report


# Available Ollama models
AVAILABLE_MODELS = [
    "llama3.1:latest",
    "llama3.1:70b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "deepseek-coder-v2:latest",
    "deepseek-coder:latest",
    "qwen2.5-coder:32b",
    "gemma2:latest",
]


def parse_txt_file(file_path: str) -> list[str]:
    """Parse a .txt file and extract requirements (one per line)."""
    if file_path is None:
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Filter empty lines and strip whitespace
    requirements = [line.strip() for line in lines if line.strip()]
    return requirements


async def run_analysis(
    requirements: list[str],
    optimization_strategy: str,
    quality_weights_mode: str,
    model: str,
) -> dict:
    """Run the ARLO analysis on requirements."""
    opt_mode = OptimizerMode.ILP if optimization_strategy == "ILP" else OptimizerMode.GREEDY
    
    weights_mode_map = {
        "Equally Important": ArchitectQualityWeightsMode.EQUALLY_IMPORTANT,
        "Inferred": ArchitectQualityWeightsMode.INFERRED,
    }
    weights_mode = weights_mode_map.get(quality_weights_mode, ArchitectQualityWeightsMode.INFERRED)
    
    ollama = OllamaService(model=model)
    try:
        architect = Architect(ollama_service=ollama)
        requirements_text = "\n".join(requirements)
        
        concerns, report = await architect.analyze(
            requirements_text=requirements_text,
            optimization_mode=opt_mode,
            quality_weights_mode=weights_mode,
        )
        
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
        await ollama.close()


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
    model: str,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, str, Optional[str]]:
    """Main analysis function for Gradio interface."""
    if file is None:
        return (
            "⚠️ Please upload a .txt file with requirements.",
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            None,
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
        )
    
    # Run analysis
    result = asyncio.run(run_analysis(requirements, optimization_strategy, quality_weights_mode, model))
    
    if not result["success"]:
        return (
            f"❌ Analysis failed: {result.get('error', 'Unknown error')}",
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            None,
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
    )


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
                    choices=AVAILABLE_MODELS,
                    value="llama3.1:latest",
                    label="Ollama Model",
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
        
        analyze_btn.click(
            fn=analyze_requirements,
            inputs=[file_input, optimization_strategy, quality_weights_mode, model_dropdown],
            outputs=[summary_output, asrs_table, decisions_table, report_output, pdf_output],
        )
    
    return app


# Create the app instance
gradio_app = create_gradio_app()
