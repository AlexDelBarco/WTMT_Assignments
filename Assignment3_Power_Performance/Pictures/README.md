
This script performs a detailed analysis of wind turbine power performance based on 
measured wind speed and power data. The analysis includes data cleaning, normalization, 
bin-averaged statistics, and the calculation of Annual Energy Production (AEP) with 
uncertainties.

Key Features:
-------------
1. **Data Import and Cleaning**:
   - Reads 10-minute averaged wind turbine data from a CSV file.
   - Removes outliers for various parameters such as wind speed, turbulence intensity, 
     pressure, temperature, humidity, pitch angle, rotor speed, and active power.
   - Filters data based on wind direction sectors.

2. **Normalization**:
   - Normalizes power and wind speed based on air density and reference conditions.
   - Calculates mean air density at the site.

3. **Power Curve Determination**:
   - Calculates bin-averaged statistics for wind speed, power, standard deviation of power, 
     Cp-coefficient, and uncertainties (category A, B, and combined).
   - Saves binned statistics to a CSV file.
   - Plots power curve and Cp as a function of wind speed.

4. **Annual Energy Production (AEP) Calculation**:
   - Calculates AEP using Rayleigh wind speed distributions for average annual wind speeds 
     (V_ave) ranging from 4 to 11 m/s.
   - Includes absolute and relative uncertainties for AEP.
   - Calculates extrapolated AEP for wind speeds outside the measured range.
   - Assigns completeness labels ("complete" or "incomplete") based on data coverage.

5. **Visualization**:
   - Generates scatter plots and error bar plots for various parameters.
   - Saves plots to a "Pictures" directory.

6. **Output**:
   - Saves AEP statistics and extrapolated power curve data to CSV files.
   - Prints bin-averaged statistics and AEP results.

Constants:
----------
- P_RATED: Rated power of the turbine (kW)
- D: Rotor diameter (m)
- HUB_HEIGHT: Hub height (m)
- WS_CUTIN: Cut-in wind speed (m/s)
- WS_CUTOUT: Cut-out wind speed (m/s)
- A: Rotor swept area (m²)
- RHO_0: Reference air density (kg/m³)
- Nh: Number of hours in a year

Usage:
------
1. Ensure the input data file (`windData.csv`) is in the same directory as this script.
2. Run the script to perform the analysis and generate results.
3. Check the "Pictures" directory for generated plots and the output CSV files for 
   statistical results.

