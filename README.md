# 🚛 ReRoute HK: Predictive Recycling Logistics

**Solving Hong Kong's "First-Mile Problem" using Spatial Data Science, Behavioral Economics, and OSRM Street Routing.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reroute-hk.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Award:** Built for the CUHK Data Hack | Advancing UN SDGs 11 (Sustainable Cities) & 12 (Responsible Consumption)

---

## 📖 The First-Mile Problem
Despite the expansion of the GREEN@COMMUNITY network, Hong Kong loses over **52.5%** of its recyclable commodities daily. Our behavioral data analysis reveals this is not an infrastructure capacity issue, but a psychological friction issue: a lack of incentive, trust, and convenience. In an ultra-dense urban environment, expecting residents (especially the elderly) to carry heavy recyclables 15 minutes to a premium station is an operational failure.

**ReRoute HK** flips the model. Instead of waiting for residents to come to the infrastructure, we use spatial data science and Operations Research to dynamically route the infrastructure directly to them.

---

## 🧠 Technical Architecture & AI Logic

Our platform operates on a robust, 5-stage data science pipeline:

### 1. Dynamic Recycling Friction Index (RFI)
We built a weighted friction model that calculates a unique score for every district based on population density, elderly demographics, and distance to baseline bins. Planners can adjust policy weights dynamically to mitigate temporal data bias.

### 2. Geospatially Snapped Weighted K-Means
Standard clustering places centroids in impossible locations (e.g., the ocean). Our algorithm weights the clustering by the RFI score, then mathematically "snaps" the coordinates to the nearest physical Public Housing Estate to guarantee real-world deployment feasibility. It proposes **1 Anchor Hub** in the highest-friction red zone.

### 3. Demographic Heuristic Time Windows
The system reads local demographics to maximize collection rates. Heavily elderly estates trigger morning deployment windows, while working-class districts trigger evening windows. Trucks are locked to strict 120-minute pop-up service durations.

### 4. Capacitated Vehicle Routing Problem (CVRPTW-IR)
Built with **Google OR-Tools**, our AI tracks physical payload limits. When a truck "cubes out", it is dynamically detoured to the nearest fixed Hub to empty, resetting its capacity mid-shift and eliminating depot return times (Continuous Replenishment).

### 5. OSRM Polyline Decoding
Euclidean "straight lines" ignore Hong Kong's complex geography. The app queries a live **Open Source Routing Machine (OSRM) API** to snap the OR-Tools node sequence to the actual street network, rendering animated, fuel-optimized dispatch paths.

---

## 💰 Financial ROI & Business Impact
By shifting from a static infrastructure model to ReRoute HK's dynamic Hub-and-Spoke deployment, the government achieves the exact same geographic coverage while realizing massive CapEx/OpEx efficiencies:
* **Taxpayer Savings:** HK$ 17.4 Million saved annually in operational costs.
* **Recoverable Value Pool:** Unlocks a massive share of the HK$ 5.04 Billion annual recyclable commodity pool (Paper, Plastics, Metals) by bypassing the 52.5% behavioral friction barrier.

---

## 📊 Open Data Sources
This project integrates multiple Hong Kong public datasets to ensure analytical rigor:
* **2021 Population Census**
* **Public Housing Database**
* **Open Space Recycling Stations**
* **Recycling Collection Points**
* **Environmental Protection Department (EPD) Recovery Statistics**

---

## 💻 How to Run Locally

To run this dashboard on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/reroute-hk.git](https://github.com/your-username/reroute-hk.git)
   cd reroute-hk
