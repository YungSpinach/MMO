import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.interpolate import interp1d # type: ignore
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
# import snowflake.connector

# FEATURES TO ADD
#----------------------------------------------
# Carbon scores
# Estimated revenue (channel budget * ROAS)
# Custom CPMs
#----------------------------------------------
# Lumen attention scores - better measure?
# Mandatory channels
# Minimum / Maximum spends for mandatory channels 
# Pareto optimisation (multi-objective)
# Changes score calculation - not optimise towards explicit: 
#  - Reach %
#  - ROAS
#  - Attention
#  - Suitability
#  - Would need to factor in diminishing returns to all of these
# Then evaluate 1000s of model combinations
# Explicitly factor in time-frames, e.g. “this is a 2 week campaign”
#---------------------------------------------- 
# Custom cover curves from Meta / YouTube / TikTok?
# See the difference between last and most recent media plans? 
# Econometric response curves for clients that have them  
# Factoring in de-duplication of reach? 
# - Overlap %s of ad channels from touchpoints?


st.image("https://images.squarespace-cdn.com/content/5c9e3048523958515c382443/2129c340-d177-48e6-8b14-3c8b01a94ec7/CreamLogo-EMAILSIGNATURE.png?content-type=image%2Fpng", width=100)
st.text("")
st.title("Media Mix Optimiser :gear:")
st.text("")
st.write("### Input Parameters - required :pencil:")
col1, col2 = st.columns(2)
audience_name = col1.selectbox("Audience Name", ["ABC1 Adults", "Women 16-34", "ABC1 Women", "ABC1 Men"])
total_budget = col2.number_input("Total Budget (£)", min_value=0, value=500000)
marketing_objective = st.selectbox("Marketing Objective", ["Salience", "Unaided Awareness", "Aided Awareness", "Association", "Consideration", "Purchase Intent"])

# ====================
# Input Data
# ====================

# Set the target audience
# audience_name = "ABC1 Adults"  # User-defined audience name

# Read cover curves data
# Snowflake connection parameters
conn = st.connection("snowflake")

# Query data from Snowflake tables
query1 = "SELECT * FROM COVER_CURVES"
query2 = "SELECT * FROM MEDIA_EFFECTIVENESS"

cover_curves_df = pd.read_sql(query1, conn)
media_effectiveness_df = pd.read_sql(query2, conn)

# Filter data for the target audience
cover_curves_df = cover_curves_df[cover_curves_df["AUDIENCE_NAME"] == audience_name]
media_effectiveness_df = media_effectiveness_df[media_effectiveness_df["AUDIENCE_NAME"] == audience_name]

# Merge CPM from media_effectiveness into cover_curves
cover_curves_df = cover_curves_df.merge(
    media_effectiveness_df[["MEDIA_CHANNEL", "CPM"]],
    on="MEDIA_CHANNEL",
    how="left"
)



# Calculate Media Investment
cover_curves_df["Media Investment"] = cover_curves_df["CPM"] * (cover_curves_df["IMPRESSIONS"] / 1000)

# Convert cover curves data into a dictionary
cover_curves = {}
for channel, group in cover_curves_df.groupby("MEDIA_CHANNEL"):
    cover_curves[channel] = {
        "Media Investment": group["Media Investment"].tolist(),
        "COVER_PERCENT": group["COVER_PERCENT"].tolist(),
        "AVG_FREQ": group["AVG_FREQ"].tolist(),
        "CPM": group["CPM"].iloc[0],  # CPM is constant per channel
    }

# Convert media effectiveness data into a dictionary
media_effectiveness = {}
for _, row in media_effectiveness_df.iterrows():
    media_effectiveness[row["MEDIA_CHANNEL"]] = {
        "SHORT_TERM_ROI": row["SHORT_TERM_ROI"],
        "FULL_ROI": row["FULL_ROI"],
        "ATTENTION": row["ATTENTION"],
        "SALIENCE": row["SALIENCE"],
        "UNAIDED_AWARENESS": row["UNAIDED_AWARENESS"],
        "AIDED_AWARENESS": row["AIDED_AWARENESS"],
        "ASSOCIATION": row["ASSOCIATION"],
        "CONSIDERATION": row["CONSIDERATION"],
        "PURCHASE_INTENT": row["PURCHASE_INTENT"],
    }

# Total budget
# total_budget = 500000  # $1,000,000

# Marketing objective (e.g., "Salience", "Unaided Awareness", etc.)
# marketing_objective = "Consideration"  # User-defined marketing objective

# Constraints
# frequency_cap = 10  # Max frequency per channel
budget_caps = {channel: 0.4 for channel in cover_curves}  # Max % of total budget per channel (example: 20%)


st.text("")
st.text("")
st.write("#### Input Parameters - optional 	:paperclip:")

col1, col2 = st.columns(2)
frequency_cap = col1.number_input("Frequency Cap", min_value=0, value=10)
max_channels = col2.number_input("Max. Channels", min_value=0, value=None)

excluded_channels = st.multiselect("Channels to exclude", ["Audio", 
                                                           "BVOD", 
                                                           "Cinema", 
                                                           "Generic Search", 
                                                           "Linear TV", 
                                                           "Magazines - Print",
                                                           "Newspapers - Pint",
                                                           "Online Display",
                                                           "Online Video",
                                                           "OOH",
                                                           "Paid Social"])

# Weights for scoring criteria (based on marketing objective)
st.text("")
st.text("Adjust weights?")
col1, col2 = st.columns(2)
weights = {
    "SHORT_TERM_ROI": col1.slider("Short-Term ROI 	:pound:", min_value=0.0, max_value=1.0, value=0.1),
    "FULL_ROI": col1.slider("Full ROI :moneybag:", min_value=0.0, max_value=1.0, value=0.4),
    "ATTENTION": col2.slider("Attention :eyes:", min_value=0.0, max_value=1.0, value=0.1),
    "Suitability": col2.slider(f"{marketing_objective}", min_value=0.0, max_value=1.0, value=0.4),
}


# ====================
# Helper Functions
# ====================

def calculate_score(channel, allocated_budget):
    """Calculate the score for a channel based on allocated budget."""
    # Interpolate cover % and frequency
    investment = cover_curves[channel]["Media Investment"]
    cover = cover_curves[channel]["COVER_PERCENT"]
    frequency = cover_curves[channel]["AVG_FREQ"]
    
    # Create interpolation functions
    cover_interp = interp1d(investment, cover, fill_value="extrapolate")
    frequency_interp = interp1d(investment, frequency, fill_value="extrapolate")
    
    # Calculate cover percentage and average frequency for the allocated budget
    cover_pct = cover_interp(allocated_budget)
    avg_frequency = frequency_interp(allocated_budget)
    
    # Calculate GRPs (Gross Rating Points)
    grps = cover_pct * avg_frequency
    
    # Get effectiveness coefficients
    short_term_roi = media_effectiveness[channel]["SHORT_TERM_ROI"]
    full_roi = media_effectiveness[channel]["FULL_ROI"]
    attention = media_effectiveness[channel]["ATTENTION"]
    suitability = media_effectiveness[channel][marketing_objective]  # Suitability depends on the marketing objective
    
    # Normalize coefficients to a range of 0 to 1
    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    
    # Calculate min and max values for each coefficient
    short_term_roi_min = min([media_effectiveness[ch]["SHORT_TERM_ROI"] for ch in media_effectiveness])
    short_term_roi_max = max([media_effectiveness[ch]["SHORT_TERM_ROI"] for ch in media_effectiveness])
    
    full_roi_min = min([media_effectiveness[ch]["FULL_ROI"] for ch in media_effectiveness])
    full_roi_max = max([media_effectiveness[ch]["FULL_ROI"] for ch in media_effectiveness])
    
    attention_min = min([media_effectiveness[ch]["ATTENTION"] for ch in media_effectiveness])
    attention_max = max([media_effectiveness[ch]["ATTENTION"] for ch in media_effectiveness])
    
    suitability_min = min([media_effectiveness[ch][marketing_objective] for ch in media_effectiveness])
    suitability_max = max([media_effectiveness[ch][marketing_objective] for ch in media_effectiveness])
    
    # Normalize the coefficients
    short_term_roi_norm = normalize(short_term_roi, short_term_roi_min, short_term_roi_max)
    full_roi_norm = normalize(full_roi, full_roi_min, full_roi_max)
    attention_norm = normalize(attention, attention_min, attention_max)
    suitability_norm = normalize(suitability, suitability_min, suitability_max)
    
    # Calculate the weighted sum of normalized coefficients
    weighted_sum = (
        weights["SHORT_TERM_ROI"] * short_term_roi_norm +
        weights["FULL_ROI"] * full_roi_norm +
        weights["ATTENTION"] * attention_norm +
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

def allocate_budget(total_budget, budget_caps, frequency_cap, max_channels=None, excluded_channels=None):
    """Allocate budget across channels iteratively."""
    # Initialize allocation for each channel to 0
    allocation = {channel: 0 for channel in cover_curves}
    remaining_budget = total_budget
    
    # If excluded_channels is provided, remove those channels from consideration
    if excluded_channels:
        for channel in excluded_channels:
            if channel in allocation:
                del allocation[channel]
    
    # Track the number of channels with non-zero allocation
    channels_allocated = 0
    
    # Iterate until the entire budget is allocated
    while remaining_budget > 0:
        best_score = -1  # Initialize the best score to a very low value
        best_channel = None  # Initialize the best channel to None
        
        # Evaluate each channel
        for channel in allocation:
            # Skip if the channel has reached its budget cap
            if allocation[channel] + 0.05 * total_budget > budget_caps[channel] * total_budget:
                continue
            
            # Skip if the maximum number of channels has been reached
            if max_channels is not None and channels_allocated >= max_channels and allocation[channel] == 0:
                continue
            
            # Calculate score for the next 5% allocation
            new_allocation = allocation[channel] + 0.05 * total_budget
            score, cover_pct, avg_frequency, grps = calculate_score(channel, new_allocation)
            
            # Skip if the frequency exceeds the cap
            if avg_frequency > frequency_cap:
                continue
            
            # Update the best channel if this channel has a higher score
            if score > best_score:
                best_score = score
                best_channel = channel
        
        # If no valid channel is found, stop the allocation process
        if best_channel is None:
            break
        
        # Allocate 5% of the budget to the best channel
        allocation[best_channel] += 0.05 * total_budget
        remaining_budget -= 0.05 * total_budget
        
        # Increment the count of allocated channels if this is a new channel
        if allocation[best_channel] == 0.05 * total_budget:
            channels_allocated += 1
    
    return allocation



# ====================
# Run the Algorithm
# ====================

# Allocate budget
allocation = allocate_budget(total_budget, budget_caps, frequency_cap, max_channels, excluded_channels)

# Generate output table
output_table = []
for channel, budget in allocation.items():
    if budget > 0:  # Only include channels with non-zero budget allocation
        score, cover_pct, avg_frequency, grps = calculate_score(channel, budget)
        full_roas = media_effectiveness[channel]["FULL_ROI"]
        estimated_revenue = budget * full_roas
        output_table.append({
            "Media Channel": channel,
            "Budget Allocation (£)": f"£{budget:,.0f}",
            "Budget Allocation (%)": f"{(budget / total_budget) * 100:.0f}%",
            "CPM (£)": f"£{cover_curves[channel]['CPM']:,.0f}",
            "Cover (%)": f"{np.round(cover_pct, 1)}%",
            "Avg. Frequency": np.round(avg_frequency, 1),
            "GRPs": np.round(grps, 1),
            "Est. ROAS": np.round(full_roas, 2),
            "Est. Revenue (£)": f"£{estimated_revenue:,.0f}",
        })

output_df = pd.DataFrame(output_table)

# ====================
# Show Results
# ====================

st.text("")
st.text("")
st.text("")
st.write("### Recommended Channel Splits :bar_chart:")
st.dataframe(output_df, hide_index=True)
st.text("")

#Exporting the table
# import csv

# Show topline callouts for 

col1, col2, col3 = st.columns(3)
col1.metric("Total Budget", f"£{total_budget:,.0f}", border=True)
col2.metric("Est. Revenue", f"£{int(output_df['Est. Revenue (£)'].str.replace('£', '').str.replace(',', '').astype(float).sum()):,.0f}", border=True)
col3.metric("Est. ROAS", f"{np.round(output_df['Est. Revenue (£)'].str.replace('£', '').str.replace(',', '').astype(float).sum() / total_budget, 2)}", border=True)


st.text("")

# ====================
# R+F Graph
# ====================

# Plot Cover % and Avg. Frequency by Channel
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar chart for Cover %
channels = output_df["Media Channel"]
cover_pct = output_df["Cover (%)"].str.rstrip('%').astype(float)
ax1.bar(channels, cover_pct, color='orangered', alpha=0.6, label='Cover %')

# Set x-axis labels
ax1.set_xticklabels(channels, rotation=45, ha='right')

# Set y-axis label for Cover %
ax1.set_ylabel('Cover %', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y')
ax1.set_ylim(0, cover_pct.max() + 10)
# Add data labels to each bar
for i, v in enumerate(cover_pct):
    ax1.text(i, v + 1, f"{v:.1f}%", color='black', ha='center', va='center')

# Create a secondary y-axis for Avg. Frequency
ax2 = ax1.twinx()
avg_frequency = output_df["Avg. Frequency"]
ax2.plot(channels, avg_frequency, color='mediumpurple', marker='o', markersize=30, alpha=0.8)

# Set y-axis label for Avg. Frequency
ax2.set_ylabel('Avg. Frequency', fontsize=14, fontweight='bold')
ax2.tick_params(axis='y')
ax2.set_ylim(0.8, avg_frequency.max() + 0.5)

# Set chart title
plt.title("Channel Reach (bars) and Avg. Frequencies (line and dots)", fontsize=20, fontweight='bold')

# Display the plot in Streamlit
st.pyplot(fig)


# ====================
# Cover Curve Graph
# ====================

st.text("")
# Plot Cover Curves by Channel
fig2, ax = plt.subplots(figsize=(12, 6))

# Define colors for each channel
colors = plt.cm.get_cmap('tab10', len(cover_curves))

# Plot cover curves for each channel
for i, (channel, data) in enumerate(cover_curves.items()):
    investment = data["Media Investment"]
    cover_pct = data["Cover %"]
    color = colors(i)
    
    # Plot the cover curve line
    ax.plot(investment, cover_pct, label=channel, color=color)
    
    # If the channel has been allocated budget, add a symbol to indicate the current allocated spend level
    if channel in allocation and allocation[channel] > 0:
        allocated_investment = allocation[channel]
        min_investment = min(investment)
        if allocated_investment >= min_investment:
            ax.scatter(allocated_investment, interp1d(investment, cover_pct, fill_value="extrapolate")(allocated_investment), color=color, s=100, zorder=5)

# Titles
ax.set_xlabel("Investment (£)", fontsize=14, fontweight='bold')
ax.set_ylabel("Cover %", fontsize=14, fontweight='bold')
ax.set_title('Channel Cover Curves, dots indicate current recommended spend level', fontsize=20, fontweight='bold')


# Set x-axis limit
ax.set_xlim(0, total_budget / 2.5)

# Add legend
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig2)

# ====================
# Metric Rankings - under the bonnet
# ====================

st.text("")
st.text("")
st.text("")
st.write("#### Appendix : Channel Scores by Metric 	:paperclip:")

#CPMs
st.text("CPMs by Channel")
st.bar_chart(media_effectiveness_df, x="MEDIA_CHANNEL", y="CPM", use_container_width=True, color="Media Channel")

col1, col2 = st.columns(2)

# Short-Term ROIs
col1.text("Short-Term ROIs by Channel")
col1.bar_chart(media_effectiveness_df, x="MEDIA_CHANNEL", y="SHORT_TERM_ROI", horizontal=True)

# Full ROIs
col2.text("Full ROIs by Channel")
col2.bar_chart(media_effectiveness_df, x="MEDIA_CHANNEL", y="FULL_ROI", horizontal=True)

# Attention
col1.text("Attention by Channel")
col1.bar_chart(media_effectiveness_df, x="MEDIA_CHANNEL", y="ATTENTION", horizontal=True)

# Suitability
col2.text(f"{marketing_objective} by Channel")
col2.bar_chart(media_effectiveness_df, x="MEDIA_CHANNEL", y=marketing_objective, horizontal=True)