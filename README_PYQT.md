# Decline Curve Analysis Tool (PyQt6 Version)

A professional petroleum engineering application for analyzing production decline curves using Arps hyperbolic decline models with Bayesian optimization.

## Features

### Analysis Capabilities
- **Automatic decline detection** using smoothing and peak analysis
- **Manual start date selection** with interactive chart picking
- **Fixed initial rate (Qi)** option for constrained fitting
- **Bayesian optimization** for parameter tuning
- **Outlier filtering** with iterative refinement
- **Production period detection** for multi-phase wells

### User Interface
- **Modern PyQt6 interface** with professional styling
- **Interactive matplotlib charts** with zoom, pan, and save
- **Resizable panels** for flexible workflow
- **Tabbed parameter controls** for organized settings
- **Real-time plot updates** when changing display options

### Visualization Options
- Show/hide outliers
- Show/hide pre-decline data points
- Toggle forecast display
- Display smoothed data
- Show uncertainty channel
- Customizable forecast duration and offset

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- PyQt6 >= 6.5.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- bayesian-optimization >= 1.4.0

## Usage

### Starting the Application
```bash
python withPYQT_17.py
```

### Workflow

1. **Select Well**
   - Choose District, Field, Formation, and Well from dropdowns
   - Application automatically filters options based on available data

2. **Configure Start Method**
   - **Auto Select**: Automatically detect decline start
   - **Manual Start Date**: Specify year and month manually
   - **Manual Start Date & Initial Rate (Qi)**: Fix both start and initial rate

3. **Set Parameters**
   - **General**: Outlier threshold, forecast settings
   - **Decline Detection**: Smoothing windows, peak detection
   - **Period Detection**: Production phase identification
   - **Plot Options**: Control what's displayed on the chart

4. **Run Analysis**
   - Click "Run Analysis" to fit the Arps model
   - View results in the text area
   - Interact with the chart using matplotlib tools

5. **Refine Parameters**
   - Enable "Optimize Parameters" for Bayesian optimization
   - Configure optimization ranges if needed
   - Check specific parameters to optimize

### Interactive Features

#### Chart Selection
- **Manual Start Date**: Click "Select" then click on the chart to pick a start date
- **Qi Selection**: Click "Select" in the Qi section, then click on the chart to pick both date and rate

#### Plot Customization
- Use checkboxes in "Plot Options" tab to toggle elements
- Changes apply immediately to existing analysis
- Adjust forecast offset to shift forecast timeline

## Data Format

The application expects a CSV file (`OFM202409.csv`) with columns:
- `District`: Geographic district
- `Field`: Oil field name
- `Alias_Formation`: Formation name
- `Well_Name`: Unique well identifier
- `Prod_Date`: Production date (datetime)
- `M_Oil_Prod`: Monthly oil production (barrels)

## Analysis Methods

### Arps Hyperbolic Model
```
q(t) = qi / (1 + b * Di * t)^(1/b)
```
Where:
- `qi`: Initial production rate (bbl/day)
- `Di`: Initial decline rate (1/month)
- `b`: Hyperbolic exponent (0-2)
- `t`: Time (months)

### Decline Start Detection
1. Apply Savitzky-Golay smoothing
2. Detect peaks with prominence filtering
3. Verify sustained decline after peak
4. Check moving average drop and slope criteria
5. Ensure persistence of decline trend

### Outlier Filtering
- Iterative process (max 10 iterations)
- Fit model to current data subset
- Calculate residuals and standard deviation
- Remove points beyond threshold × std
- Protect trailing data points from removal
- Converge when no more outliers detected

### Bayesian Optimization
- Maximizes fit quality score combining:
  - R² (coefficient of determination)
  - MAPE (mean absolute percentage error)
  - Decline reasonableness
  - Data utilization
  - Trend consistency
- 5 initial random points + 15 optimization iterations (auto select)
- 5 initial + 10 iterations (manual start)

## Keyboard Shortcuts

- **Ctrl+R**: Run Analysis (if implemented)
- **Ctrl+Q**: Quit Application (if implemented)

## Troubleshooting

### Common Issues

**"Please select a well"**
- Ensure all dropdowns are populated
- Check that data file exists and is valid

**"Invalid manual date"**
- Verify year and month are within data range
- Format: YYYY for year, 1-12 for month

**"Please enter an Initial Rate (Qi)"**
- Required when using "Manual Start Date & Initial Rate (Qi)"
- Either type value or use chart selection

**Optimization failed**
- Check parameter ranges in configuration
- Ensure ranges are valid (min < max)
- Verify data quality (enough points)

### Performance Tips
1. Disable optimization for faster initial analysis
2. Limit optimizable parameters to essentials
3. Use narrower optimization ranges for faster convergence

## Development

### Project Structure
```
withPYQT_17.py          # Main application
requirements.txt         # Python dependencies
UPGRADE_NOTES.md        # Migration details
UI_COMPARISON.md        # Visual improvements
README_PYQT.md          # This file
```

### Extending the Application

#### Adding New Parameters
1. Add to DEFAULT_*_RANGES dictionaries
2. Create QLineEdit and QCheckBox in appropriate tab
3. Add to parameter extraction in run_analysis()
4. Update param_to_opt_check mapping
5. Add to reset_parameters()

#### Custom Styling
Modify the setStyleSheet() in __init__():
```python
self.setStyleSheet("""
    /* Your custom CSS-like styles */
""")
```

## Credits

**Original tkinter version**: WithGUI_Grok_16.py  
**PyQt6 conversion**: 2025

## License

[Your License Here]

## Support

For issues or questions:
1. Check this README
2. Review UPGRADE_NOTES.md
3. Examine UI_COMPARISON.md for styling info

## Version History

**v17.0 (PyQt6)**
- Complete UI overhaul with PyQt6
- Modern styling and color scheme
- Improved layout with resizable panels
- Better visual feedback and interactions
- Enhanced user experience

**v16.0 (tkinter)**
- Final tkinter-based version
- Full Bayesian optimization
- Interactive chart selection
- Multiple analysis modes

