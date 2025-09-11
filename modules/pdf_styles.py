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
                font-size: 8pt;
                color: #666;
            }
        }
        
        /* Print-specific adjustments */
        @media print {
            body {
                font-size: 8pt;
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
            font-size: 8pt;
            line-height: 1.3;
            color: #000;
            background: white;
            width: 180mm;
            box-sizing: border-box;
            margin: 0 auto;
            padding: 0;
        }
        
        /* Headers and titles - matching app hierarchy */
        .pdf-title {
            font-size: 12pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 12pt;
            color: #2c5282;
            padding-bottom: 6pt;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .pdf-subtitle {
            font-size: 10pt;
            font-weight: bold;
            margin-bottom: 10pt;
            color: #2c5282;
            text-align: center;
            padding: 6pt 0;
            background-color: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }
        
        .pdf-section-header {
            font-size: 9pt;
            font-weight: bold;
            margin-top: 12pt;
            margin-bottom: 6pt;
            color: #2c5282;
            padding-bottom: 4pt;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* Dataset information box - matching app style */
        .pdf-dataset-info {
            background-color: #f8f9fa;
            border: 1px solid #e2e8f0;
            padding: 6pt;
            margin-bottom: 8pt;
            border-radius: 4px;
            page-break-inside: avoid;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .pdf-dataset-info h3 {
            margin-top: 0;
            margin-bottom: 6pt;
            font-size: 9pt;
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
            padding: 4pt 6pt;
            border: 1px solid #e2e8f0;
            border-radius: 3px;
            width: 25%;
            vertical-align: top;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .pdf-info-label {
            font-weight: bold;
            font-size: 6pt;
            color: #666;
            display: block;
            margin-bottom: 2pt;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .pdf-info-value {
            font-size: 8pt;
            color: #2d3748;
            font-weight: 500;
        }
        
        /* Tables - matching app style */
        .pdf-table {
            width: 100%;
            border-collapse: collapse;
            margin: 8pt 0;
            font-size: 8pt;
            page-break-inside: auto;
            background-color: white;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            table-layout: fixed;
        }
        
        .pdf-table th {
            background-color: #f5f5f5;
            padding: 4pt 6pt;
            text-align: left;
            border: 1px solid #e2e8f0;
            font-weight: bold;
            font-size: 8pt;
            color: #2d3748;
        }
        
        .pdf-table td {
            padding: 4pt 6pt;
            border: 1px solid #e2e8f0;
            vertical-align: top;
            font-size: 8pt;
            color: #2d3748;
        }

        /* Column width tuning for personalised analysis tables */
        .pdf-table th:nth-child(1),
        .pdf-table td:nth-child(1) { /* Risk Level */
            width: 9%;
        }
        .pdf-table th:nth-child(2),
        .pdf-table td:nth-child(2) { /* Potential Progression */
            width: 27%;
        }
        .pdf-table th:nth-child(3),
        .pdf-table td:nth-child(3) { /* Expected Timeline */
            width: 20%;
            white-space: normal;
        }
        .pdf-table th:nth-child(4),
        .pdf-table td:nth-child(4) { /* Statistical Support */
            width: 14%;
        }
        .pdf-table th:nth-child(5),
        .pdf-table td:nth-child(5) { /* Progression Details */
            width: 30%;
            overflow-wrap: break-word;
            word-break: break-word;
        }
        
        .pdf-table tr {
            page-break-inside: avoid;
        }
        
        /* Condition sections - matching app style */
        .pdf-condition-section {
            margin-bottom: 10pt;
            page-break-inside: avoid;
            border: 1px solid #e2e8f0;
            padding: 6pt;
            background-color: #f8f9fa;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .pdf-condition-header {
            font-size: 9pt;
            font-weight: bold;
            color: #2c5282;
            margin-bottom: 6pt;
            padding-bottom: 4.5pt;
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
            font-size: 6pt;
        }
        
        /* Risk indicators for personalised analysis (badge-style to match app) */
        .pdf-risk-high,
        .pdf-risk-moderate,
        .pdf-risk-low {
            display: inline-block;
            padding: 2pt 6pt;
            border-radius: 3pt;
            font-weight: normal;
            font-size: 8pt;
            line-height: 1;
            margin: 0;
        }

        /* System tag styling to match app in PDFs */
        .system-tag {
            display: inline-block;
            padding: 1pt 4pt;
            border-radius: 3pt;
            background-color: #e2e8f0;
            color: #2d3748;
            font-size: 8pt;
            white-space: nowrap;
        }

        .pdf-risk-high {
            background-color: #dc3545;
            color: #ffffff;
        }

        .pdf-risk-moderate {
            background-color: #ffc107;
            color: #ffffff;
        }

        .pdf-risk-low {
            background-color: #28a745;
            color: #ffffff;
        }
        
        /* Footer */
        .pdf-footer {
            margin-top: 12pt;
            padding-top: 6pt;
            border-top: 1px solid #ccc;
            font-size: 6pt;
            color: #666;
            text-align: center;
        }
        
        /* Responsive adjustments for smaller content */
        @media print {
            .pdf-table {
                font-size: 8pt;
            }
            
            .pdf-table th,
            .pdf-table td {
                padding: 4pt 6pt;
                font-size: 8pt;
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