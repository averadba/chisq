import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.power import GofChisquarePower

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to calculate expected frequencies and decide the test
def calculate_test(table):
    chi2, p, dof, expected = chi2_contingency(table)
    test_used = ""
    if np.all(expected >= 5):
        test_result = f"Chi-square test:\nChi2 Statistic: {chi2:.4f}, p-value: {p:.4f}"
        test_used = "Chi-square test"
        effect_size = (chi2 / np.sum(table))
    else:
        oddsratio, p = fisher_exact(table)
        test_result = f"Fisher's Exact Test:\np-value: {p:.4f}, Odds Ratio: {oddsratio:.4f}"
        test_used = "Fisher's Exact Test"
        effect_size = np.sqrt((np.log(oddsratio))**2 / np.sum(table))

    # Power analysis
    power_analysis = GofChisquarePower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs=np.sum(table), alpha=0.05, n_bins=2**2)
    
    # Returning both test results and expected frequencies
    expected_df = pd.DataFrame(expected, columns=table_input.columns, index=table_input.index)
    return test_result, power, expected_df, test_used

# Streamlit UI
st.title('2x2 Contingency Table Analysis')
st.markdown("**By:** [Dr. Alexis Vera](mailto:alexisvera@gmail.com)")

# Instructions
st.write("Please fill in all numbers in the contingency table and press 'Compute'.")
st.write("If expected frequencies are less than 5, a Fisher's Exact Test will be performed instead of a Chi-square test.")

# Setting up a default DataFrame for the contingency table
default_data = pd.DataFrame([[0, 0], [0, 0]], columns=['Column 1', 'Column 2'], index=['Row 1', 'Row 2'])

# Using experimental_data_editor for user input
table_input = st.data_editor(default_data, height=150, key="input_table")

# Convert DataFrame to numpy array
table = table_input.to_numpy()

# Button for computation
compute_clicked = st.button('Compute')

# Validate and compute
if compute_clicked:
    if np.any(pd.isnull(table)) or not np.issubdtype(table.dtype, np.number):
        st.error('Please ensure all entries are numeric and the table is completely filled.')
    else:
        result, power, expected_df, test_used = calculate_test(table)
        st.write(result)
        st.write(f"Statistical Power: {power:.4f}")
        
        # Display tables side by side
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Data")
            st.dataframe(table_input)
        with col2:
            st.write("Expected Frequencies")
            st.dataframe(expected_df)

# Expander with explanation
if compute_clicked:
    with st.expander("Learn More About This Analysis"):
        st.write("""
        ### Purpose of This App
        This app performs statistical tests on a 2x2 contingency table provided by the user. Depending on the data, it either conducts a Chi-square test or a Fisher's Exact test to determine if there are non-random associations between the variables in the table.

        ### How to Interpret Results
        - **Chi-square Test**: Used when expected frequencies are 5 or more. The p-value indicates whether the observed association could be random (p > 0.05) or not (p < 0.05).
        - **Fisher's Exact Test**: Used for smaller sample sizes or when expected frequencies are less than 5. It provides an exact p-value and odds ratio, measuring the strength of the association.
        - **Statistical Power**: Indicates the probability that the test correctly rejects the null hypothesis when it is false. Higher power reduces the risk of Type II errors.

        ### Statistical Calculations
        - **Chi-square Test**: Calculated as χ² = Σ((O-E)²/E) where O is the observed frequency and E is the expected frequency.
        - **Fisher's Exact Test**: Calculates the exact probability of observing the data assuming the null hypothesis of no effect is true.
        - **Statistical Power (for Chi-square)**: Computed using the formula involving effect size, sample size, significance level, and degrees of freedom.
        """)
