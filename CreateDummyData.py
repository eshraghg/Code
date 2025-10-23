import pandas as pd
import numpy as np
from datetime import datetime
import calendar

# Arps hyperbolic function from the original code
def arps_hyperbolic(t, qi, Di, b):
    return qi / (1 + b * Di * t) ** (1/b)

# Function to get days in month
def get_days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

# Define hierarchy: 2 Districts, 2 Fields each, 2 Formations each, 2 Wells each -> 16 wells
districts = ['District1', 'District2']
fields_per_district = 2
formations_per_field = 2
wells_per_formation = 2

# Parameters ranges for random generation
qi_range = (500, 2000)  # Initial rate bbl/day
Di_range = (0.05, 0.5)  # Initial decline rate
b_range = (0.5, 1.5)    # Hyperbolic factor
months = 60             # 5 years of monthly data

# Generate start dates varying between 2015 and 2020
start_year_range = (2015, 2020)

# List to hold all data rows
data = []

# Generate data
for dist_idx, district in enumerate(districts):
    for field_idx in range(fields_per_district):
        field = f"Field_{dist_idx+1}_{field_idx+1}"
        for form_idx in range(formations_per_field):
            formation = f"Formation_{dist_idx+1}_{field_idx+1}_{form_idx+1}"
            for well_idx in range(wells_per_formation):
                well = f"Well_{dist_idx+1}_{field_idx+1}_{form_idx+1}_{well_idx+1}"
                
                # Random parameters for this well
                qi = np.random.uniform(*qi_range)
                Di = np.random.uniform(*Di_range)
                b = np.random.uniform(*b_range)
                
                # Random start date: first of a random month in random year
                start_year = np.random.randint(*start_year_range)
                start_month = np.random.randint(1, 13)
                start_date = datetime(start_year, start_month, 1)
                
                # Generate time array (months)
                t = np.arange(months)
                
                # Daily rates
                daily_rates = arps_hyperbolic(t, qi, Di, b)
                
                # Add some noise for realism (5-10% variation)
                noise = np.random.normal(1, 0.075, months)
                daily_rates *= noise
                daily_rates = np.maximum(daily_rates, 0)  # No negative rates
                
                # Generate dates and monthly production
                for i in range(months):
                    current_date = start_date + pd.DateOffset(months=i)
                    days = get_days_in_month(current_date)
                    m_oil_prod = daily_rates[i] * days
                    
                    # Append row
                    data.append({
                        'District': district,
                        'Field': field,
                        'Alias_Formation': formation,
                        'Well_Name': well,
                        'Prod_Date': current_date.strftime('%Y-%m-%d'),
                        'M_Oil_Prod': round(m_oil_prod, 2)
                    })

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Well_Name and Prod_Date for cleanliness
df = df.sort_values(['Well_Name', 'Prod_Date'])

# Save to CSV
df.to_csv('OFM202409.csv', index=False)

print(f"Generated dummy data for {len(df['Well_Name'].unique())} wells with {months} months each.")
print("CSV file 'OFM202409.csv' created successfully.")