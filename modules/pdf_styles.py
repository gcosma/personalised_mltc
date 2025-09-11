"""
PDF-optimized CSS styles for A4 format exports.

This module contains CSS templates optimized for A4 printing and PDF generation.
"""

def get_pdf_base_styles():
    """Base CSS styles optimized for A4 PDF printing"""
    return """
    <style>
        /* A4 Page Setup */
        @page {
            size: A4;
            margin: 15mm;
            @top-right {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 8px;
                color: #666;
            }
        }
        
        /* Print-specific adjustments */
        @media print {
            body {
                font-size: 8px;
                line-height: 1.3;
                color: #000;
                background: white;
            }
            
            * {
                -webkit-print-color-adjust: exact !important;
                color-adjust: exact !important;
                print-color-adjust: exact !important;
            }
            
            /* Hide instruction banner when printing */
            .pdf-print-instructions {
                display: none !important;
            }
        }
        
        /* Base layout for PDF */
        .pdf-container {
            font-family: 'Times New Roman', Times, serif;
            font-size: 8px;
            line-height: 1.3;
            color: #000;
            background: white;
            max-width: 180mm; /* A4 width minus smaller margins */
            margin: 0 auto;
            padding: 0;
        }
        
        /* Headers and titles - matching app hierarchy */
        .pdf-title {
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #2c5282;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .pdf-subtitle {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c5282;
            text-align: center;
            padding: 8px 0;
            background-color: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }
        
        .pdf-section-header {
            font-size: 10px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 8px;
            color: #2c5282;
            padding-bottom: 4px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* Dataset information box - matching app style */
        .pdf-dataset-info {
            background-color: #f8f9fa;
            border: 1px solid #e2e8f0;
            padding: 12px;
            margin-bottom: 15px;
            border-radius: 6px;
            page-break-inside: avoid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .pdf-dataset-info h3 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 10px;
            color: #2c5282;
            font-weight: bold;
        }
        
        .pdf-info-grid {
            display: table;
            width: 100%;
            border-spacing: 6px;
        }
        
        .pdf-info-row {
            display: table-row;
        }
        
        .pdf-info-item {
            display: table-cell;
            background-color: white;
            padding: 6px 8px;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            width: 25%;
            vertical-align: top;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .pdf-info-label {
            font-weight: bold;
            font-size: 6px;
            color: #666;
            display: block;
            margin-bottom: 2px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .pdf-info-value {
            font-size: 8px;
            color: #2d3748;
            font-weight: 500;
        }
        
        /* Tables - matching app style */
        .pdf-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 7px;
            page-break-inside: auto;
            background-color: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .pdf-table th {
            background-color: #f5f5f5;
            padding: 6px 8px;
            text-align: left;
            border: 1px solid #e2e8f0;
            font-weight: bold;
            font-size: 6px;
            color: #2d3748;
        }
        
        .pdf-table td {
            padding: 5px 8px;
            border: 1px solid #e2e8f0;
            vertical-align: top;
            font-size: 6px;
            color: #2d3748;
        }
        
        .pdf-table tr {
            page-break-inside: avoid;
        }
        
        /* Condition sections - matching app style */
        .pdf-condition-section {
            margin-bottom: 20px;
            page-break-inside: avoid;
            border: 1px solid #e2e8f0;
            padding: 12px;
            background-color: #f8f9fa;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .pdf-condition-header {
            font-size: 9px;
            font-weight: bold;
            color: #2c5282;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        /* Page breaks */
        .pdf-page-break {
            page-break-before: always;
        }
        
        .pdf-avoid-break {
            page-break-inside: avoid;
        }
        
        /* Text formatting */
        .pdf-emphasis {
            font-weight: bold;
            color: #000;
        }
        
        .pdf-muted {
            color: #666;
            font-size: 6px;
        }
        
        /* Risk indicators for personalised analysis */
        .pdf-risk-high {
            background-color: #ffe6e6;
            border-left: 3px solid #dc3545;
            padding: 3px;
        }
        
        .pdf-risk-moderate {
            background-color: #fff3cd;
            border-left: 3px solid #ffc107;
            padding: 3px;
        }
        
        .pdf-risk-low {
            background-color: #d4edda;
            border-left: 3px solid #28a745;
            padding: 3px;
        }
        
        /* Footer */
        .pdf-footer {
            margin-top: 15px;
            padding-top: 8px;
            border-top: 1px solid #ccc;
            font-size: 6px;
            color: #666;
            text-align: center;
        }
        
        /* Responsive adjustments for smaller content */
        @media print {
            .pdf-table {
                font-size: 6px;
            }
            
            .pdf-table th,
            .pdf-table td {
                padding: 2px 1px;
                font-size: 5px;
            }
            
            .pdf-info-item {
                font-size: 6px;
            }
        }
        
        /* Hide interactive elements */
        .pdf-hide {
            display: none !important;
        }
    </style>
    """

def get_network_pdf_styles():
    """Additional styles specifically for network visualization PDFs"""
    return """
    <style>
        /* Network-specific PDF styles */
        .pdf-network-container {
            page-break-inside: avoid;
            margin-bottom: 20pt;
        }
        
        .pdf-network-summary {
            background-color: #f5f5f5;
            border: 1pt solid #ccc;
            padding: 10pt;
            margin-bottom: 16pt;
            border-radius: 2pt;
        }
        
        .pdf-network-legend {
            background-color: white;
            border: 1pt solid #ddd;
            padding: 8pt;
            margin-bottom: 12pt;
        }
        
        .pdf-network-legend h4 {
            margin-top: 0;
            margin-bottom: 6pt;
            font-size: 10pt;
            font-weight: bold;
        }
        
        .pdf-legend-item {
            display: flex;
            align-items: center;
            margin: 2pt 0;
            font-size: 9pt;
        }
        
        .pdf-legend-color {
            width: 12pt;
            height: 12pt;
            margin-right: 6pt;
            border: 1pt solid #999;
            display: inline-block;
        }
        
        /* Network statistics table */
        .pdf-network-stats {
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0;
            font-size: 9pt;
        }
        
        .pdf-network-stats th,
        .pdf-network-stats td {
            padding: 4pt;
            border: 1pt solid #ccc;
            text-align: left;
        }
        
        .pdf-network-stats th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
    </style>
    """

def get_pdf_print_instructions():
    """Clear PDF printing instructions banner - hidden when printing"""
    return """
    <div class="pdf-print-instructions" style="position: fixed; top: 15px; right: 15px; z-index: 9999; 
                background: #2c5282; color: white; padding: 16px 20px; border-radius: 8px; 
                box-shadow: 0 4px 12px rgba(44, 82, 130, 0.25); border: 1px solid #1a365d;
                font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; 
                max-width: 300px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 18px; margin-right: 8px;">📄</span>
            <strong style="font-size: 15px; color: white;">Save as PDF</strong>
        </div>
        <div style="font-size: 13px; line-height: 1.5; margin-bottom: 12px; color: #e2e8f0;">
            Right-click anywhere → <strong style="color: white;">Print</strong><br>
            Or press <strong style="color: white;">Ctrl+P</strong> (Windows) / <strong style="color: white;">Cmd+P</strong> (Mac)
        </div>
        <div style="font-size: 12px; color: #cbd5e0; border-top: 1px solid #4a5568; 
                    padding-top: 10px;">
            💡 Choose "Save as PDF" as destination
        </div>
    </div>
    """

def get_print_instructions():
    """Fallback instructions - now empty since we use the new banner"""
    return ""