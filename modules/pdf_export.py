"""
PDF Export module for DECODE application.

This module handles the conversion of analysis results to PDF-optimized HTML format.
Since external PDF libraries aren't available, this generates HTML optimized for 
browser-based PDF printing.
"""

import re
from datetime import datetime
from modules.pdf_styles import (
    get_pdf_base_styles, 
    get_network_pdf_styles, 
    get_print_instructions,
    get_pdf_print_instructions
)
from modules.config import condition_categories

def build_document_title(export_type, dataset_info, analysis_params=None):
    """Build a descriptive document title used by the browser as the default PDF filename."""
    if analysis_params is None:
        analysis_params = {}

    export_title_map = {
        'personalised_analysis': 'Personalised Trajectory',
        'network_viz': 'Cohort Network',
        'network_graph': 'Custom Trajectory Network'
    }
    base = export_title_map.get(export_type, 'DECODE Analysis')

    parts = ["DECODE", base]

    # Dataset bits
    db = dataset_info.get('database') if dataset_info else None
    gender = dataset_info.get('gender') if dataset_info else None
    age = dataset_info.get('age_group') if dataset_info else None
    dataset_bits = " ".join(bit for bit in [db, gender, age] if bit and bit != 'Unknown')
    if dataset_bits:
        parts.append(dataset_bits)

    # Analysis params
    min_or = analysis_params.get('min_or')
    if min_or is not None:
        parts.append(f"OR≥{min_or}")

    min_freq = analysis_params.get('min_freq')
    if min_freq is not None:
        parts.append(f"min n={min_freq}")

    time_horizon = analysis_params.get('time_horizon')
    if time_horizon is not None:
        parts.append(f"Time {time_horizon}y")

    sel = analysis_params.get('selected_conditions') or []
    if sel:
        shown = sel[:2]
        if len(sel) > 2:
            shown_title = ", ".join(shown) + f" (+{len(sel)-2} more)"
        else:
            shown_title = ", ".join(shown)
        parts.append(shown_title)

    return " — ".join(parts)

def generate_pdf_html(content_html, export_type, dataset_info, analysis_params=None):
    """
    Convert regular HTML content to PDF-optimized HTML
    
    Args:
        content_html: Original HTML content
        export_type: Type of export ('personalised_analysis', 'network_viz', 'network_graph')
        dataset_info: Dictionary with database, gender, age_group
        analysis_params: Analysis parameters for header info
    
    Returns:
        PDF-optimized HTML string
    """
    # Get appropriate styles
    base_styles = get_pdf_base_styles()
    if export_type in ['network_viz', 'network_graph']:
        additional_styles = get_network_pdf_styles()
    else:
        additional_styles = ""
    
    # Generate header with dataset information
    header_html = generate_pdf_header(dataset_info, analysis_params, export_type)
    
    # Convert content to PDF format
    if export_type == 'personalised_analysis':
        pdf_content = convert_personalised_analysis_to_pdf(content_html)
    elif export_type == 'network_viz':
        pdf_content = convert_network_viz_to_pdf(content_html, dataset_info, analysis_params)
    elif export_type == 'network_graph':
        pdf_content = convert_network_graph_to_pdf(content_html, dataset_info, analysis_params)
    else:
        pdf_content = convert_generic_content_to_pdf(content_html)
    
    # Generate footer
    footer_html = generate_pdf_footer()
    
    # Build document title for better default PDF filename
    page_title = build_document_title(export_type, dataset_info, analysis_params)

    # Combine everything
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{page_title}</title>
        {base_styles}
        {additional_styles}
    </head>
    <body>
        {get_print_instructions()}
        {get_pdf_print_instructions()}
        <div class="pdf-container">
            {header_html}
            {pdf_content}
            {footer_html}
        </div>
    </body>
    </html>
    """
    
    return full_html

def generate_pdf_header(dataset_info, analysis_params, export_type):
    """Generate PDF header with dataset and analysis information"""
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    
    # Format export type title
    title_map = {
        'personalised_analysis': 'Personalised Trajectory Analysis',
        'network_viz': 'Cohort Network Analysis', 
        'network_graph': 'Custom Trajectory Network Analysis'
    }
    title = title_map.get(export_type, 'DECODE Analysis')
    
    header_html = f"""
    <div class="pdf-avoid-break">
        <h1 class="pdf-title">DECODE-MIDAS Analysis Report</h1>
        <h2 class="pdf-subtitle">{title}</h2>
        
        <div class="pdf-dataset-info">
            <h3>📊 Dataset & Analysis Information</h3>
            <div class="pdf-info-grid">
                <div class="pdf-info-row">
                    <div class="pdf-info-item">
                        <span class="pdf-info-label">DATABASE</span>
                        <span class="pdf-info-value">{dataset_info.get('database', 'Unknown')}</span>
                    </div>
                    <div class="pdf-info-item">
                        <span class="pdf-info-label">POPULATION</span>
                        <span class="pdf-info-value">{dataset_info.get('gender', 'Unknown')} {dataset_info.get('age_group', 'Unknown')}</span>
                    </div>
                    <div class="pdf-info-item">
                        <span class="pdf-info-label">GENERATED</span>
                        <span class="pdf-info-value">{timestamp}</span>
                    </div>
                    <div class="pdf-info-item">
                        <span class="pdf-info-label">EXPORT TYPE</span>
                        <span class="pdf-info-value">{title}</span>
                    </div>
                </div>
            </div>"""
    
    # Add analysis-specific parameters if available
    if analysis_params:
        param_items = []
        if 'min_or' in analysis_params:
            param_items.append(f'<div class="pdf-info-item"><span class="pdf-info-label">MIN ODDS RATIO</span><span class="pdf-info-value">{analysis_params["min_or"]}</span></div>')
        if 'min_freq' in analysis_params:
            param_items.append(f'<div class="pdf-info-item"><span class="pdf-info-label">MIN FREQUENCY</span><span class="pdf-info-value">{analysis_params["min_freq"]}</span></div>')
        if 'time_horizon' in analysis_params:
            param_items.append(f'<div class="pdf-info-item"><span class="pdf-info-label">TIME HORIZON</span><span class="pdf-info-value">{analysis_params["time_horizon"]} years</span></div>')
        if 'selected_conditions' in analysis_params and analysis_params['selected_conditions']:
            conditions_str = ', '.join(analysis_params['selected_conditions'][:3])  # Limit to first 3 for space
            if len(analysis_params['selected_conditions']) > 3:
                conditions_str += f" (and {len(analysis_params['selected_conditions']) - 3} more)"
            param_items.append(f'<div class="pdf-info-item"><span class="pdf-info-label">CONDITIONS</span><span class="pdf-info-value">{conditions_str}</span></div>')
        
        if param_items:
            header_html += f"""
            <div class="pdf-info-grid">
                <div class="pdf-info-row">
                    {"".join(param_items[:4])}
                </div>
            </div>"""
    
    header_html += """
        </div>
    </div>
    """
    
    return header_html

def convert_personalised_analysis_to_pdf(html_content):
    """Convert personalised analysis HTML to PDF format"""
    # Extract the content from the HTML
    # Remove browser-specific styling and convert to PDF classes
    
    # Remove the original styling and container divs
    content = re.sub(r'<style>.*?</style>', '', html_content, flags=re.DOTALL)
    content = re.sub(r'<div[^>]*class="patient-analysis"[^>]*>', '<div class="pdf-container">', content)
    content = re.sub(r'<div[^>]*class="analysis-container"[^>]*>', '', content)
    content = re.sub(r'<div[^>]*class="condition-section"[^>]*>', '<div class="pdf-condition-section pdf-avoid-break">', content)
    content = re.sub(r'<div[^>]*class="condition-header"[^>]*>', '<h3 class="pdf-condition-header">', content)
    content = re.sub(r'<div[^>]*class="summary-section"[^>]*>', '<div class="pdf-dataset-info pdf-avoid-break">', content)
    
    # Convert tables
    content = re.sub(r'<table[^>]*class="trajectory-table"[^>]*>', '<table class="pdf-table">', content)
    
    # Convert risk indicators
    content = re.sub(r'style="background-color:\s*#dc3545[^"]*"', 'class="pdf-risk-high"', content)
    content = re.sub(r'style="background-color:\s*#ffc107[^"]*"', 'class="pdf-risk-moderate"', content)
    content = re.sub(r'style="background-color:\s*#28a745[^"]*"', 'class="pdf-risk-low"', content)
    # Remove duplicate class attributes introduced by replacement (keep PDF class)
    content = re.sub(r'class="risk-badge"\s+class="(pdf-risk-(?:high|moderate|low))"', r'class="\1"', content)
    content = re.sub(r'class="(pdf-risk-(?:high|moderate|low))"\s+class="risk-badge"', r'class="\1"', content)
    
    # Remove duplicate elements since we're adding them to the header
    # Remove the dataset info section and all its components
    content = re.sub(r'<div[^>]*class="dataset-info"[^>]*>.*?</div>', '', content, flags=re.DOTALL)
    
    # Also remove individual info items that might escape the container removal
    content = re.sub(r'<div[^>]*class="info-grid"[^>]*>.*?</div>', '', content, flags=re.DOTALL)
    content = re.sub(r'<div[^>]*class="info-item"[^>]*>.*?</div>', '', content, flags=re.DOTALL)
    
    # Remove specific dataset information patterns by label
    content = re.sub(r'<span[^>]*class="info-label"[^>]*>(POPULATION|TOTAL PATIENTS|MIN ODDS RATIO|TIME HORIZON|DATABASE)</span>.*?</div>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove the duplicate main title "Personalised Disease Trajectory Analysis" 
    content = re.sub(r'<h2[^>]*>Personalised Disease Trajectory Analysis</h2>', '', content)
    
    # Clean up any remaining inline styles that might interfere with PDF
    content = re.sub(r'style="[^"]*"', '', content)
    
    return content

def convert_network_viz_to_pdf(html_content, dataset_info, analysis_params):
    """Convert network visualization to PDF format with summary"""
    # Since network visualizations are interactive and don't translate well to PDF,
    # create a summary representation instead
    # Also remove any duplicate dataset information from the original content
    
    pdf_content = f"""
    <div class="pdf-network-container">
        <h2 class="pdf-section-header">Network Analysis Summary</h2>
        
        <div class="pdf-network-summary">
            <p><strong>Analysis Type:</strong> Cohort Network Visualization</p>
            <p><strong>Note:</strong> This is a summary representation of an interactive network visualization. 
               The original interactive network shows relationships between medical conditions as nodes connected by edges, 
               where edge thickness indicates association strength and colors represent body systems.</p>
        </div>
        
        <div class="pdf-network-legend">
            <h4>Body System Categories</h4>
            <div class="pdf-legend-items">
    """
    
    # Add body system legend
    for system, color in condition_categories.items():
        if system != "Other":
            pdf_content += f"""
                <div class="pdf-legend-item">
                    <span class="pdf-legend-color" style="background-color: {color}50; border-color: {color};"></span>
                    <span>{system}</span>
                </div>
            """
    
    pdf_content += f"""
            </div>
        </div>
        
        <div class="pdf-network-summary">
            <h4>Analysis Parameters</h4>
            <table class="pdf-network-stats">
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Minimum Odds Ratio</td>
                    <td>{analysis_params.get('min_or', 'N/A')}</td>
                    <td>Threshold for including condition relationships</td>
                </tr>
                <tr>
                    <td>Minimum Frequency</td>
                    <td>{analysis_params.get('min_freq', 'N/A')}</td>
                    <td>Minimum number of patient occurrences required</td>
                </tr>
                <tr>
                    <td>Database</td>
                    <td>{dataset_info.get('database', 'N/A')}</td>
                    <td>Source database for analysis</td>
                </tr>
                <tr>
                    <td>Population</td>
                    <td>{dataset_info.get('gender', 'N/A')} {dataset_info.get('age_group', 'N/A')}</td>
                    <td>Demographic subset analyzed</td>
                </tr>
            </table>
        </div>
        
        <div class="pdf-muted">
            <p><em>For the complete interactive network visualization, please refer to the HTML export or view in the application.</em></p>
        </div>
    </div>
    """
    
    return pdf_content

def convert_network_graph_to_pdf(html_content, dataset_info, analysis_params):
    """Convert network graph to PDF format with summary"""  
    # Similar to network viz but with trajectory-specific information
    # Remove any duplicate dataset information from the original content
    
    conditions = analysis_params.get('selected_conditions', [])
    conditions_str = ', '.join(conditions) if conditions else 'None specified'
    
    pdf_content = f"""
    <div class="pdf-network-container">
        <h2 class="pdf-section-header">Custom Trajectory Network Summary</h2>
        
        <div class="pdf-network-summary">
            <p><strong>Analysis Type:</strong> Personalized Trajectory Network</p>
            <p><strong>Initial Conditions:</strong> {conditions_str}</p>
            <p><strong>Note:</strong> This analysis shows potential disease progression pathways from selected initial conditions. 
               The interactive network displays conditions as nodes with directed edges indicating likely progression patterns.</p>
        </div>
        
        <div class="pdf-network-summary">
            <h4>Analysis Configuration</h4>
            <table class="pdf-network-stats">
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Initial Conditions</td>
                    <td>{len(conditions)}</td>
                    <td>{conditions_str}</td>
                </tr>
                <tr>
                    <td>Minimum Odds Ratio</td>
                    <td>{analysis_params.get('min_or', 'N/A')}</td>
                    <td>Threshold for progression relationships</td>
                </tr>
                <tr>
                    <td>Time Horizon</td>
                    <td>{analysis_params.get('time_horizon', 'N/A')} years</td>
                    <td>Maximum time period for projections</td>
                </tr>
                <tr>
                    <td>Minimum Frequency</td>
                    <td>{analysis_params.get('min_freq', 'N/A')}</td>
                    <td>Minimum patient occurrences for relationships</td>
                </tr>
            </table>
        </div>
        
        <div class="pdf-muted">
            <p><em>The complete trajectory network with interactive features is available in the HTML export or application view.</em></p>
        </div>
    </div>
    """
    
    return pdf_content

def convert_generic_content_to_pdf(html_content):
    """Convert generic HTML content to PDF format"""
    # Basic cleanup for any other content types
    content = re.sub(r'<style>.*?</style>', '', html_content, flags=re.DOTALL)
    
    # Remove duplicate dataset information that might appear in legends or side panels
    content = re.sub(r'<div[^>]*>.*?Dataset Information.*?</div>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<div[^>]*>.*?<strong>Database:</strong>.*?</div>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up inline styles
    content = re.sub(r'style="[^"]*"', '', content)
    return f'<div class="pdf-avoid-break">{content}</div>'

def generate_pdf_footer():
    """Generate PDF footer"""
    return f"""
    <div class="pdf-footer">
        <p>Generated by DECODE-MIDAS: Multimorbidity in Intellectual Disability Analysis System</p>
        <p>© 2024 DECODE Project, Loughborough University | Funded by NIHR</p>
    </div>
    """