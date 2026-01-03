"""PDF report generator for ARLO."""
import os
import tempfile
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML


def get_template_path() -> str:
    """Get the path to the templates directory."""
    return os.path.join(os.path.dirname(__file__), "templates")


def generate_pdf_report(analysis_result: dict) -> str:
    """
    Generate a PDF report from analysis results.
    
    Args:
        analysis_result: Dictionary containing analysis results with keys:
            - concerns: List of Concern objects
            - asrs: List of ASR objects
            - requirements: List of Requirement objects
            - report: Text report string
    
    Returns:
        Path to the generated PDF file.
    """
    # Prepare template data
    template_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_requirements": len(analysis_result.get("requirements", [])),
        "asr_count": len(analysis_result.get("asrs", [])),
        "concern_count": len(analysis_result.get("concerns", [])),
        "asrs": [],
        "concerns": [],
    }
    
    # Process ASRs
    for asr in analysis_result.get("asrs", []):
        template_data["asrs"].append({
            "id": asr.id,
            "description": asr.description,
            "quality_attributes": ", ".join(asr.quality_attributes) if asr.quality_attributes else "-",
            "condition": asr.condition_text or "-",
        })
    
    # Process Concerns and Decisions
    for i, concern in enumerate(analysis_result.get("concerns", []), 1):
        concern_data = {
            "index": i,
            "conditions": concern.conditions,
            "desired_qualities": concern.desired_qualities,
            "average_score": f"{concern.average_score:.2f}",
            "total_score": concern.total_score,
            "decisions": [],
        }
        
        for decision in concern.decisions:
            decision_data = {
                "pattern_type": decision.arch_pattern_name,
                "selected_pattern": decision.selected_pattern,
                "score": decision.score,
                "satisfied": [{"quality": q, "score": s} for q, s in decision.satisfied_qualities],
                "tradeoffs": [{"quality": q, "score": s} for q, s in decision.unsatisfied_qualities],
            }
            concern_data["decisions"].append(decision_data)
        
        template_data["concerns"].append(concern_data)
    
    # Load and render template
    env = Environment(loader=FileSystemLoader(get_template_path()))
    template = env.get_template("report.html")
    html_content = template.render(**template_data)
    
    # Generate PDF
    pdf_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        prefix="arlo_report_",
    )
    pdf_file.close()
    
    HTML(string=html_content).write_pdf(pdf_file.name)
    
    return pdf_file.name
