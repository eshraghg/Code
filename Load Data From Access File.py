
# Import required libraries
import pyodbc
import pandas as pd

# Show all columns in pandas output
pd.set_option('display.max_columns', None)


# Function to read tables from Access .mdb file
def read_mdb_database(mdb_file_path):
    """Read specific tables from .mdb Access database"""
    conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_file_path};'
    # List of tables to read
    tables_to_read = ["Production", "Test", "WaterInjection", "XYCoordinate", "GasInjection"]
    tables_dict = {}
    try:
        conn = pyodbc.connect(conn_str)
        # Read each table into a DataFrame
        for table_name in tables_to_read:
            try:
                df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
                tables_dict[table_name] = df
                print(f"✓ {table_name}: {len(df)} rows")
            except:
                print(f"✗ {table_name}: not found")
        conn.close()
        return tables_dict
    except Exception as e:
        print(f"Database error: {e}")
        return None



# Read tables from Access database file
mdb_file = r"Z:\SHF\Code\ofm\OFM202409.mdb"
tables = read_mdb_database(mdb_file)

# Select and filter columns for each table
df_Production = tables['Production']

# Select relevant columns from Test table
df_test = tables['Test']
df_test = df_test[['Well_Name','Test_Date', 'Choke', 'WHP', 'BHP', 'T_GOR', 'T_PDOIL', 'Api', 'T_WC',
    'S_GOR', 'MFP', 'S_Press', 'S_Temp', 'LGR']]

# Select relevant columns from WaterInjection table
df_WaterInjection = tables['WaterInjection']
df_WaterInjection = df_WaterInjection[['Well_Name', 'Inject_Date', 'DaysOnInject', 'I_Choke', 'I_Press', 'M_Water_Inject',
    'IDWINJ', 'CDWINJ_C', 'CDWINJ_P']]

# Select relevant columns from XYCoordinate table
df_XYCoordinate = tables['XYCoordinate']
df_XYCoordinate = df_XYCoordinate[['District', 'Field', 'Well_Name']]

# Select relevant columns from GasInjection table
df_GasInjection = tables['GasInjection']
df_GasInjection = df_GasInjection[['Well_Name','Inject_Date',
    'DaysOnInject', 'I_Choke', 'I_Press', 'M_Gas_Inject', 'CDGIWN', 'CGIWN',
    'CDGIIOOC', 'CGIIOOC', 'CDGID', 'CGID', 'CDGIF', 'CGIF', 'CDGIR',
    'CGIR', 'IDGINJ', 'CDGINJ_C', 'CDGINJ_P']]

# Merge XYCoordinate and Production tables on Well_Name
df_all = df_XYCoordinate.merge(df_Production, on='Well_Name', how='inner')

mdb_file_name = mdb_file.split('\\')[-1]
csv_file_name = mdb_file_name.replace('.mdb', '.csv')
df_all.to_csv(csv_file_name, index=False, header=True)
