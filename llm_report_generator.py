"""
LLM Report Generator for Decline Curve Analysis

This module provides functionality to generate natural language reports
from decline curve analysis results using Ollama with Llama models.
"""

import ollama
from typing import Dict, List, Optional, Any
import traceback


class LLMReportGenerator:
    """Generate natural language reports from DCA analysis results"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the LLM report generator
        
        Args:
            model_name: Name of the Ollama model to use (default: llama3.2:3b)
        """
        self.model_name = model_name
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # Try to list models to check connection
            models_response = ollama.list()
            
            # Extract models list - Ollama returns a ListResponse object with a 'models' attribute
            models_list = []
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            elif isinstance(models_response, list):
                models_list = models_response
            
            # Extract model names safely
            model_names = []
            for m in models_list:
                # Ollama Model objects use 'model' attribute, not 'name'
                if hasattr(m, 'model'):
                    model_names.append(str(m.model))
                elif hasattr(m, 'name'):
                    model_names.append(str(m.name))
                elif isinstance(m, dict):
                    # Fallback for dict format
                    name = m.get('model') or m.get('name') or m.get('model_name')
                    if name:
                        model_names.append(str(name))
            
            # Check if requested model is available
            if self.model_name not in model_names:
                print(f"Warning: Model '{self.model_name}' not found. Available models: {model_names}")
                print(f"You may need to run: ollama pull {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Please ensure Ollama is installed and running.")
            print("Download from: https://ollama.ai")
    
    def extract_analysis_data(self, results: str, model_data: Dict, 
                             well_name: str, reference_models: Optional[List[Dict]] = None,
                             start_method: str = "") -> Dict[str, Any]:
        """
        Extract structured data from analysis results for LLM prompt
        
        Args:
            results: Raw results text string
            model_data: Dictionary containing model parameters and data
            well_name: Name of the well being analyzed
            reference_models: List of reference model dictionaries (optional)
            start_method: Method used to determine start of decline
            
        Returns:
            Dictionary containing structured analysis data
        """
        extracted = {
            'well_name': well_name,
            'start_method': start_method,
            'parameters': {},
            'forecast': {},
            'data_quality': {},
            'references': []
        }
        
        # Extract parameters from model_data
        if model_data and 'popt' in model_data:
            popt = model_data['popt']
            if len(popt) >= 3:
                extracted['parameters'] = {
                    'qi': round(popt[0], 2),
                    'Di': round(popt[1], 6),
                    'b': round(popt[2], 2)
                }
        
        # Extract forecast data
        if model_data:
            if 'forecast_dates' in model_data and 'forecast_q' in model_data:
                forecast_dates = model_data['forecast_dates']
                forecast_q = model_data['forecast_q']
                if forecast_q and len(forecast_q) > 0:
                    initial_rate = forecast_q[0]
                    final_rate = forecast_q[-1] if len(forecast_q) > 1 else initial_rate
                    forecast_months = len(forecast_q)
                    decline_percent = ((initial_rate - final_rate) / initial_rate * 100) if initial_rate > 0 else 0
                    
                    extracted['forecast'] = {
                        'initial_rate': round(initial_rate, 2),
                        'final_rate': round(final_rate, 2),
                        'duration_months': forecast_months,
                        'duration_years': round(forecast_months / 12, 1),
                        'decline_percent': round(decline_percent, 1)
                    }
        
        # Extract data quality metrics from results string
        lines = results.split('\n')
        for line in lines:
            line = line.strip()
            if 'Coefficient of Variation' in line or 'CV:' in line:
                try:
                    cv_value = line.split(':')[-1].strip()
                    cv = float(cv_value)
                    extracted['data_quality']['cv'] = round(cv, 3)
                    if cv > 0.4:
                        extracted['data_quality']['quality'] = 'Poor'
                        extracted['data_quality']['warning'] = 'High variability detected. Forecast may be unreliable.'
                    elif cv > 0.3:
                        extracted['data_quality']['quality'] = 'Moderate'
                        extracted['data_quality']['warning'] = 'Moderate variability. Review forecast carefully.'
                    else:
                        extracted['data_quality']['quality'] = 'Good'
                        extracted['data_quality']['warning'] = None
                except:
                    pass
            elif 'outliers removed' in line.lower():
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        outlier_part = parts[-1].strip()
                        # Extract number before "out of"
                        outlier_count = outlier_part.split(' out of')[0].strip()
                        extracted['data_quality']['outliers_removed'] = int(outlier_count.split()[-1])
                except:
                    pass
        
        # Extract reference models
        if reference_models:
            for ref in reference_models:
                ref_data = {
                    'name': ref.get('name', 'Unknown'),
                    'Di': ref.get('Di', 0),
                    'b': ref.get('b', 0),
                    'well_name': ref.get('well_name', 'Unknown')
                }
                extracted['references'].append(ref_data)
        
        return extracted
    
    def build_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """
        Build a structured prompt for the LLM
        
        Args:
            analysis_data: Extracted analysis data dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are an expert petroleum engineer specializing in decline curve analysis (DCA) using Arps hyperbolic decline models. Your task is to generate a professional technical report analyzing the provided decline curve analysis results.

"""
        
        # Analysis summary section
        prompt += "=== ANALYSIS DATA ===\n\n"
        prompt += f"Well/Selection: {analysis_data.get('well_name', 'Unknown')}\n"
        prompt += f"Analysis Method: {analysis_data.get('start_method', 'Not specified')}\n\n"
        
        # Parameters
        params = analysis_data.get('parameters', {})
        if params:
            prompt += "Fitted Arps Parameters:\n"
            prompt += f"  - Initial Rate (qi): {params.get('qi', 0)} bbl/day\n"
            prompt += f"  - Decline Rate (Di): {params.get('Di', 0)} /day ({params.get('Di', 0)*365:.4f} /year)\n"
            prompt += f"  - Decline Exponent (b): {params.get('b', 0)}\n\n"
        
        # Forecast
        forecast = analysis_data.get('forecast', {})
        if forecast:
            prompt += "Forecast Summary:\n"
            prompt += f"  - Initial Forecast Rate: {forecast.get('initial_rate', 0)} bbl/day\n"
            prompt += f"  - Final Forecast Rate: {forecast.get('final_rate', 0)} bbl/day\n"
            prompt += f"  - Forecast Duration: {forecast.get('duration_months', 0)} months ({forecast.get('duration_years', 0)} years)\n"
            prompt += f"  - Total Decline: {forecast.get('decline_percent', 0)}%\n\n"
        
        # Data quality
        quality = analysis_data.get('data_quality', {})
        if quality:
            prompt += "Data Quality Metrics:\n"
            if 'cv' in quality:
                prompt += f"  - Coefficient of Variation (CV): {quality.get('cv', 0)}\n"
            if 'quality' in quality:
                prompt += f"  - Quality Rating: {quality.get('quality', 'Unknown')}\n"
            if 'outliers_removed' in quality:
                prompt += f"  - Outliers Removed: {quality.get('outliers_removed', 0)}\n"
            if quality.get('warning'):
                prompt += f"  - Warning: {quality.get('warning')}\n"
            prompt += "\n"
        
        # References
        references = analysis_data.get('references', [])
        if references:
            prompt += "=== REFERENCE MODELS FOR COMPARISON ===\n\n"
            for i, ref in enumerate(references, 1):
                prompt += f"Reference {i} - {ref.get('name', 'Unknown')}:\n"
                prompt += f"  - Decline Rate (Di): {ref.get('Di', 0)} /day ({ref.get('Di', 0)*365:.4f} /year)\n"
                prompt += f"  - Decline Exponent (b): {ref.get('b', 0)}\n\n"
        
        # Instructions
        prompt += """=== TASK ===

Generate a professional technical report with the following sections:

1. EXECUTIVE SUMMARY
   - Brief overview of the analysis results
   - Key findings and forecast summary

2. PARAMETER INTERPRETATION
   - Explain what the fitted parameters (qi, Di, b) mean
   - Interpret the decline rate and exponent values
   - Discuss the type of decline pattern (hyperbolic, exponential, harmonic)

3. FORECAST ANALYSIS
   - Analyze the forecast trajectory
   - Discuss production decline over the forecast period
   - Comment on forecast reliability based on data quality

4. DATA QUALITY ASSESSMENT
   - Evaluate the coefficient of variation and what it means
   - Discuss data reliability and any concerns
   - Note any outliers removed and their impact

5. REFERENCE COMPARISON"""
        
        if references:
            prompt += """
   - Compare the current analysis with reference models
   - Highlight differences in decline parameters
   - Discuss whether the current well follows similar decline patterns
   - Note any significant deviations or similarities
"""
        else:
            prompt += """
   - No reference models provided for comparison.
"""
        
        prompt += """
6. RECOMMENDATIONS
   - Provide actionable insights based on the analysis
   - Suggest any improvements or considerations
   - Note any limitations or areas of concern

Format the report professionally with clear sections. Use technical terminology appropriately. Be concise but thorough. Write in third person or passive voice as appropriate for technical reports.
"""
        
        return prompt
    
    def generate_report(self, analysis_data: Dict[str, Any], 
                       timeout: int = 120) -> str:
        """
        Generate a natural language report from analysis data
        
        Args:
            analysis_data: Extracted analysis data dictionary
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Generated report text
        """
        try:
            prompt = self.build_prompt(analysis_data)
            
            # Call Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 2000  # Max tokens for response
                }
            )
            
            report = response.get('response', '').strip()
            
            if not report:
                return "Error: LLM returned empty response. Please check your Ollama installation and model."
            
            return report
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}\n\n"
            error_msg += "Troubleshooting:\n"
            error_msg += "1. Ensure Ollama is installed and running\n"
            error_msg += f"2. Ensure model '{self.model_name}' is available: ollama pull {self.model_name}\n"
            error_msg += "3. Check your internet connection (for first-time model downloads)\n"
            error_msg += f"4. Error details: {traceback.format_exc()}"
            return error_msg

    def test_connection(self):
        """
        Test connection to Ollama and model availability
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            models_response = ollama.list()
            
            # Extract models list - Ollama returns a ListResponse object with a 'models' attribute
            models_list = []
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            elif isinstance(models_response, list):
                models_list = models_response
            
            # Extract model names safely
            model_names = []
            for m in models_list:
                # Ollama Model objects use 'model' attribute, not 'name'
                if hasattr(m, 'model'):
                    model_names.append(str(m.model))
                elif hasattr(m, 'name'):
                    model_names.append(str(m.name))
                elif isinstance(m, dict):
                    # Fallback for dict format
                    name = m.get('model') or m.get('name') or m.get('model_name')
                    if name:
                        model_names.append(str(name))
            
            if self.model_name in model_names:
                # Try a simple generation to verify it works
                test_response = ollama.generate(
                    model=self.model_name,
                    prompt="Say 'OK' if you can read this.",
                    options={'num_predict': 10}
                )
                return True, f"Connection successful. Model '{self.model_name}' is available."
            else:
                available_str = ', '.join(model_names[:5]) if model_names else 'none found'
                return False, f"Model '{self.model_name}' not found. Available models: {available_str}. Run: ollama pull {self.model_name}"
        except Exception as e:
            return False, f"Cannot connect to Ollama: {str(e)}. Please ensure Ollama is installed and running."
