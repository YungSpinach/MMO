import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import streamlit as st


st.title("Media Mix Optimiser")
st.write("### Input Parameters")
col1, col2 = st.columns(2)
audience_name = col1.text_input("Audience Name", "ABC1 Adults")
total_budget = col2.number_input("Total Budget", min_value=0, value=500000)
marketing_objective = col1.selectbox("Marketing Objective", ["Salience", "Unaided Awareness", "Aided Awareness", "Association", "Consideration", "Purchase Intent"])
frequency_cap = col2.number_input("Frequency Cap", min_value=0, value=10)

# ====================
# Input Data
# ====================

# Set the target audience
# audience_name = "ABC1 Adults"  # User-defined audience name

# Read cover curves data
cover_curves_df = pd.read_csv("cover_curves.csv")
media_effectiveness_df = pd.read_csv("media_effectiveness.csv")

# Filter data for the target audience
cover_curves_df = cover_curves_df[cover_curves_df["Audience Name"] == audience_name]
media_effectiveness_df = media_effectiveness_df[media_effectiveness_df["Audience Name"] == audience_name]

# Merge CPM from media_effectiveness into cover_curves
cover_curves_df = cover_curves_df.merge(
    media_effectiveness_df[["Media Channel", "CPM"]],
    on="Media Channel",
    how="left"
)



# Calculate Media Investment
cover_curves_df["Media Investment"] = cover_curves_df["CPM"] * (cover_curves_df["Impressions"] / 1000)

# Convert cover curves data into a dictionary
cover_curves = {}
for channel, group in cover_curves_df.groupby("Media Channel"):
    cover_curves[channel] = {
        "Media Investment": group["Media Investment"].tolist(),
        "Cover %": group["Cover %"].tolist(),
        "Avg. Frequency": group["Avg. Freq"].tolist(),
        "CPM": group["CPM"].iloc[0],  # CPM is constant per channel
    }

# Convert media effectiveness data into a dictionary
media_effectiveness = {}
for _, row in media_effectiveness_df.iterrows():
    media_effectiveness[row["Media Channel"]] = {
        "Short-Term ROI": row["Short-Term ROI"],
        "Full ROI": row["Full ROI"],
        "Attention": row["Attention"],
        "Salience": row["Salience"],
        "Unaided Awareness": row["Unaided Awareness"],
        "Aided Awareness": row["Aided Awareness"],
        "Association": row["Association"],
        "Consideration": row["Consideration"],
        "Purchase Intent": row["Purchase Intent"],
    }

# Total budget
# total_budget = 500000  # $1,000,000

# Marketing objective (e.g., "Salience", "Unaided Awareness", etc.)
# marketing_objective = "Consideration"  # User-defined marketing objective

# Constraints
# frequency_cap = 10  # Max frequency per channel
budget_caps = {channel: 0.3 for channel in cover_curves}  # Max % of total budget per channel (example: 20%)



# Weights for scoring criteria (based on marketing objective)
if marketing_objective in ["Salience", "Unaided Awareness", "Aided Awareness", "Association", "Consideration", "Purchase Intent"]:
    weights = {
        "Short-Term ROI": 0.1,
        "Full ROI": 0.2,
        "Attention": 0.3,
        "Suitability": 0.4,  # Suitability weight depends on the marketing objective
    }
else:
    raise ValueError("Invalid marketing objective. Choose from: Salience, Unaided Awareness, Aided Awareness, Association, Consideration, Purchase Intent.")



# ====================
# Helper Functions
# ====================

def calculate_score(channel, allocated_budget):
    """Calculate the score for a channel based on allocated budget."""
    # Interpolate cover % and frequency
    investment = cover_curves[channel]["Media Investment"]
    cover = cover_curves[channel]["Cover %"]
    frequency = cover_curves[channel]["Avg. Frequency"]
    
    # Create interpolation functions
    cover_interp = interp1d(investment, cover, fill_value="extrapolate")
    frequency_interp = interp1d(investment, frequency, fill_value="extrapolate")
    
    # Calculate cover percentage and average frequency for the allocated budget
    cover_pct = cover_interp(allocated_budget)
    avg_frequency = frequency_interp(allocated_budget)
    
    # Calculate GRPs (Gross Rating Points)
    grps = cover_pct * avg_frequency
    
    # Get effectiveness coefficients
    short_term_roi = media_effectiveness[channel]["Short-Term ROI"]
    full_roi = media_effectiveness[channel]["Full ROI"]
    attention = media_effectiveness[channel]["Attention"]
    suitability = media_effectiveness[channel][marketing_objective]  # Suitability depends on the marketing objective
    
    # Normalize coefficients to a range of 0 to 1
    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    
    # Calculate min and max values for each coefficient
    short_term_roi_min = min([media_effectiveness[ch]["Short-Term ROI"] for ch in media_effectiveness])
    short_term_roi_max = max([media_effectiveness[ch]["Short-Term ROI"] for ch in media_effectiveness])
    
    full_roi_min = min([media_effectiveness[ch]["Full ROI"] for ch in media_effectiveness])
    full_roi_max = max([media_effectiveness[ch]["Full ROI"] for ch in media_effectiveness])
    
    attention_min = min([media_effectiveness[ch]["Attention"] for ch in media_effectiveness])
    attention_max = max([media_effectiveness[ch]["Attention"] for ch in media_effectiveness])
    
    suitability_min = min([media_effectiveness[ch][marketing_objective] for ch in media_effectiveness])
    suitability_max = max([media_effectiveness[ch][marketing_objective] for ch in media_effectiveness])
    
    # Normalize the coefficients
    short_term_roi_norm = normalize(short_term_roi, short_term_roi_min, short_term_roi_max)
    full_roi_norm = normalize(full_roi, full_roi_min, full_roi_max)
    attention_norm = normalize(attention, attention_min, attention_max)
    suitability_norm = normalize(suitability, suitability_min, suitability_max)
    
    # Calculate the weighted sum of normalized coefficients
    weighted_sum = (
        weights["Short-Term ROI"] * short_term_roi_norm +
        weights["Full ROI"] * full_roi_norm +
        weights["Attention"] * attention_norm +
        weights["Suitability"] * suitability_norm
    )
    
    # Calculate incremental cover percentage for an additional 5% investment
    incremental_budget = 0.05 * total_budget
    new_allocated_budget = allocated_budget + incremental_budget
    new_cover_pct = cover_interp(new_allocated_budget)
    incremental_cover_pct = new_cover_pct - cover_pct
    
    # Multiply the weighted sum by the incremental cover percentage
    score = weighted_sum * incremental_cover_pct
    
    return score, cover_pct, avg_frequency, grps

def allocate_budget(total_budget, budget_caps, frequency_cap):
    """Allocate budget across channels iteratively."""
    allocation = {channel: 0 for channel in cover_curves}
    remaining_budget = total_budget
    
    while remaining_budget > 0:
        best_score = -1
        best_channel = None
        
        # Evaluate each channel
        for channel in cover_curves:
            if allocation[channel] + 0.05 * total_budget > budget_caps[channel] * total_budget:
                continue  # Skip if budget cap is reached
            
            # Calculate score for the next 5% allocation
            new_allocation = allocation[channel] + 0.05 * total_budget
            score, cover_pct, avg_frequency, grps = calculate_score(channel, new_allocation)
            
            # Check frequency constraint
            if avg_frequency > frequency_cap:
                continue  # Skip if frequency exceeds cap
            
            # Update best channel
            if score > best_score:
                best_score = score
                best_channel = channel
        
        if best_channel is None:
            break  # No valid allocation left
        
        # Allocate 5% of the budget to the best channel
        allocation[best_channel] += 0.05 * total_budget
        remaining_budget -= 0.05 * total_budget
    
    return allocation



# ====================
# Run the Algorithm
# ====================

# Allocate budget
allocation = allocate_budget(total_budget, budget_caps, frequency_cap)

# Generate output table
output_table = []
for channel, budget in allocation.items():
    if budget > 0:  # Only include channels with non-zero budget allocation
        score, cover_pct, avg_frequency, grps = calculate_score(channel, budget)
        output_table.append({
            "Media Channel": channel,
            "Budget Allocation (£)": budget,
            "CPM (£)": cover_curves[channel]["CPM"],
            "Cover (%)": np.round(cover_pct, 1),
            "Avg. Frequency": np.round(avg_frequency, 1),
            "GRPs": np.round(grps, 1),
        })

output_df = pd.DataFrame(output_table)
st.dataframe(output_df, hide_index=True,)

#Exporting the table
# import csv
