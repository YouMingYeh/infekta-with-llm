---
marp: true
theme: default
paginate: true
---

<style>
section {
  font-size: 20px;
  padding: 20px;
  padding-left: 40px;
  line-height: 1.2;
}
</style>

# Deep Dive: LLM Integration & Parameter Analysis

## Agent-Based Epidemic Simulation Code Walkthrough

**Detailed Explanation of LLM Integration and Key Parameters**  
Youming Yeh • 2025-02-16

---

# Overview of Simulation Configuration

- **Time & Iterations:**

  - `TICKS_PER_DAY = 288` (5-minute intervals)
  - `NUM_DAYS = 120` simulation days

- **Information Sharing:**

  - `INFORMATION_PROB = 0.1`: 10% chance for agents to share news

- **LLM Inference:**

  - `LLM_INFERENCE_PROB = 0.1`: 10% chance to trigger LLM calls for reflection or planning

- **Epidemic Transmission Parameters:**

  - `ALPHA = 0.1`: Base transmission rate (S → E)
  - `BETA = 0.4`: Chance E becomes asymptomatic infected (IA)
  - `GAMMA = 0.3`: Chance IA becomes seriously infected (IS)
  - `THETA = 0.4`: Chance IS becomes critically infected (IC)
  - `PHI = 0.1`: Death rate in IC
  - `OMEGA = 0.1`: Recovery to immunity (M) probability

- **Time in Each State:**
  - Fixed durations (5 days for E, IA, IS, IC, R)

---

# Behavioral and Contact Parameters

- **Behavior Multipliers:**

  - `wearing_mask`: 0.1 (reduces exposure)
  - `maintaining_social_distance`: 0.3
  - `self_isolating`: 0 (no exposure)

- **Location Contact Rates:**

  - Home: 1, Bus: 5, Workplace: 3, Market: 4, School: 2, Terminal: 3

- **Exposure Calculation:**
  - Combined effects of both agents’ behaviors and location-specific contact rates

---

# Data Models for Cognitive Simulation

- **Memory Model:**

  - Attributes: `description`, `day`, `importance`

- **Behavior Model:**

  - Booleans: `wearing_mask`, `maintaining_social_distance`, `self_isolating`

- **Belief & Beliefs:**

  - Belief: `description`, `sentiment`
  - Beliefs: A list of Belief objects

- **News Model:**
  - `headline` string

---

# LLM Integration Functions

### 1. Generate Beliefs (Reflection)

- **Function:** `generate_beliefs(memories, persona)`
- **Process:**
  - Serializes recent memories (JSON)
  - Constructs a prompt including:
    - Agent's persona (demographics, personality, health, behavior)
    - JSON data of recent memories
  - Calls LLM (using `ollama.chat`) with model `"llama3.2:1b"`
  - Expects a JSON response strictly matching the Beliefs schema

---

# LLM Integration Functions (Cont.)

### 2. Generate Behavior (Planning)

- **Function:** `generate_behavior(beliefs, persona)`
- **Process:**
  - Serializes current beliefs (JSON)
  - Constructs a prompt that asks:
    - "Based on your current beliefs and epidemic context, decide daily behavior (mask, distancing, isolation)"
  - Returns behavior plan in JSON per the Behavior schema

---

# LLM for News Generation

- **Function:** `generate_news(information)`
- **Purpose:**
  - Generate a concise news headline based on epidemic stats (new infections, deaths, etc.)
- **Mechanism:**
  - Constructs a prompt with public health context
  - Calls the LLM and expects a JSON with a single key: `headline`
- **Integration:**
  - Headlines are spread to a subset of agents (10% daily), influencing their memories

---

# Agent Class: Cognitive & Epidemic Dynamics

- **Initialization:**

  - Each agent is instantiated with:
    - ID, age, gender, home, destination, routine, network neighbors, and a detailed persona string

- **Memory & Reflection:**

  - Daily interactions are summarized and stored as memories
  - `reflect(day)`:
    - Filters recent memories (last day)
    - With probability `LLM_INFERENCE_PROB`, calls `generate_beliefs`
    - Updates internal belief list

- **Behavior Planning:**

  - `plan()`:
    - With probability `LLM_INFERENCE_PROB`, calls `generate_behavior` based on current beliefs
    - Updates daily behavior which affects future interactions

- **State Update:**
  - Uses epidemic parameters (ALPHA, BETA, GAMMA, THETA, PHI, OMEGA) to update health state
  - Exposure is recalculated each tick based on interactions and behavior multipliers

---

# Social Network & Information Flow

- **Network Construction:**

  - Uses `networkx` to build strong ties (home, workplace) and weak ties (random friendships)
  - Stores network in SQLite for persistence

- **Information Sharing:**

  - Agents share news with neighbors based on `INFORMATION_PROB`
  - Received news becomes part of their memories, influencing subsequent LLM reflection

- **Interaction Dynamics:**
  - Agents are grouped by location each tick (Home, Bus, Workplace, etc.)
  - Pairwise interactions adjust exposure and potential infection events

---

# Main Simulation Loop

- **Daily Cycle:**

  - For each tick, agents interact based on current location (adjusted by behavior)
  - End-of-day: each agent summarizes interactions into a memory
  - News is generated (with LLM) and shared among agents
  - Agents reflect and plan via LLM calls
  - Health state updates occur based on cumulative exposure and transition probabilities

- **Data Tracking:**
  - Overall statistics computed (Susceptible, Exposed, Infected, etc.)
  - Network visualization at simulation end

---

# Summary & Research Implications

- **LLM Integration:**

  - Dual-step process: Reflection (memories → beliefs) and Planning (beliefs → behavior)
  - Provides a cognitive layer that adapts based on both internal and external (news) factors

- **Parameter Synergy:**

  - Epidemic transmission parameters (ALPHA, BETA, etc.) combined with behavioral modifiers (mask, distancing)
  - Social network dynamics and information sharing amplify adaptive behavior

- **Research Applications:**
  - Enables exploration of human-like decision making in epidemic spread
  - Offers insights for designing targeted public health interventions

---

# Questions & Discussion

- **LLM Refinement:**

  - How can the reflection and planning prompts be optimized?
  - What additional context could improve belief generation accuracy?

- **Parameter Sensitivity:**
  - Which parameters most significantly affect epidemic outcomes?
  - Ideas for calibration using real-world data?
