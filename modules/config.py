"""
Configuration module for the DECODE app.

Stores all static configuration and constants.
"""

# Individual dataset files
INDIVIDUAL_CSV_FILES = [
    'CPRD_Females_45to64.csv',
    'CPRD_Females_65plus.csv', 
    'CPRD_Females_below45.csv',
    'CPRD_Males_45to64.csv',
    'CPRD_Males_65plus.csv',
    'CPRD_Males_below45.csv',
    'SAIL_Females_45to64.csv',
    'SAIL_Females_65plus.csv',
    'SAIL_Females_below45.csv',    
    'SAIL_Males_45to64.csv',
    'SAIL_Males_65plus.csv',
    'SAIL_Males_below45.csv'
    ]

# Disease category mappings
condition_categories = {
    "Anaemia": "Blood",
    "Cardiac Arrhythmias": "Circulatory",
    "Coronary Heart Disease": "Circulatory",
    "Heart Failure": "Circulatory",
    "Hypertension": "Circulatory",
    "Peripheral Vascular Disease": "Circulatory",
    "Stroke": "Circulatory",
    "Barretts Oesophagus": "Digestive",
    "Chronic Constipation": "Digestive",
    "Chronic Diarrhoea": "Digestive",
    "Cirrhosis": "Digestive",
    "Dysphagia": "Digestive",
    "IBD": "Digestive",
    "Reflux Disorders": "Digestive",
    "Hearing Loss": "Ear",
    "Addisons Disease": "Endocrine",
    "Diabetes": "Endocrine",
    "Polycystic Ovary Syndrome": "Endocrine",
    "Thyroid Disorders": "Endocrine",
    "Visual Impairment": "Eye",
    "CKD": "Genitourinary",
    "Menopausal And Perimenopausal": "Genitourinary",
    
    "Dementia": "Mental",
    "Mental Illness": "Mental",
    "Tourette": "Mental",
    "Chronic Arthritis": "Musculoskeletal",
    "Chronic Pain Conditions": "Musculoskeletal",
    "Osteoporosis": "Musculoskeletal",
    "Cancer": "Neoplasms",
    "Cerebral Palsy": "Nervous",
    "Epilepsy": "Nervous",
    "Insomnia": "Nervous",
    "Multiple Sclerosis": "Nervous",
    "Neuropathic Pain": "Nervous",
    "Parkinsons": "Nervous",
    "Bronchiectasis": "Respiratory",
    "Chronic Airway Diseases": "Respiratory",
    "Chronic Pneumonia": "Respiratory",
    "Interstitial Lung Disease": "Respiratory",
    "Psoriasis": "Skin"
}


SYSTEM_COLORS = {
    "Blood": "#DC143C",        # Crimson
    "Circulatory": "#FF4500",  # Orange Red
    "Digestive": "#32CD32",    # Lime Green
    "Ear": "#4169E1",         # Royal Blue
    "Endocrine": "#BA55D3",    # Medium Orchid
    "Eye": "#1E90FF",         # Dodger Blue (changed from teal)
    "Genitourinary": "#DAA520", # Goldenrod
    "Mental": "#8B4513",       # Saddle Brown
    "Musculoskeletal": "#4682B4", # Steel Blue
    "Neoplasms": "#800080",    # Purple
    "Nervous": "#FFD700",      # Gold
    "Respiratory": "#006400",   # Dark Green (changed from teal)
    "Skin": "#F08080",        # Light Coral
    "Other": "#808080"         # Gray
}
