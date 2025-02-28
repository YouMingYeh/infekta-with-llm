# %%
import pandas as pd
import sqlite3
import networkx as nx
import random
import matplotlib.pyplot as plt
from pydantic import BaseModel
from collections import defaultdict
import math
import json
import concurrent.futures
from concurrent.futures import TimeoutError
from google import genai


# Configure Gemini API

client = genai.Client(api_key="AIzaSyDoPNkcgg2GseLnpGP7MiFkZA3gwjQNXL0")

# -------------------------------------------------------------------------------------------
# Configuration and Constants
# -------------------------------------------------------------------------------------------
TICKS_PER_DAY = 288  # 5-minute intervals per day
NUM_DAYS = 120  # Number of days to simulate
INFORMATION_PROB = (
    0.1  # Probability that an agent will share information with another agent
)

# Probability of running LLM inferences
LLM_INFERENCE_PROB = 0.1  # 10% chance to call the LLM each time

# Epidemic transmission parameters
ALPHA = 0.1  # Transmission rate from Susceptible to Exposed
BETA = 0.4  # Probability that Exposed becomes asymptomatic Infected
GAMMA = 0.3  # Probability that asymptomatic Infected becomes seriously Infected
THETA = 0.4  # Probability that seriously Infected becomes critically Infected
PHI = 0.1  # Death rate for Critically-Infected individuals
OMEGA = 0.1  # Immune rate for Recovered individuals

# Time periods individuals must stay in each state (in days)
TIME_PERIODS = {
    "E": 5,  # Exposed
    "IA": 5,  # Asymptomatic Infected
    "IS": 5,  # Seriously Infected
    "IC": 5,  # Critically Infected
    "R": 5,  # Recovered
}

# Agent states
STATES = {
    "S": "Susceptible",
    "E": "Exposed",
    "IA": "Asymptomatic Infected",
    "IS": "Seriously Infected",
    "IC": "Critically Infected",
    "R": "Recovered",
    "M": "Immune",
    "D": "Dead",
}

# Behavior effect multipliers
BEHAVIORS = {
    "wearing_mask": 0.1,
    "maintaining_social_distance": 0.3,
    "self_isolating": 0.0,
}

# Mean contact values for each location type
mean_contact_values = {
    "Home": 1,
    "Bus": 5,
    "Workplace": 3,
    "Market": 4,
    "School": 2,
    "Terminal": 3,
}


# %%
# -------------------------------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------------------------------
class Memory(BaseModel):
    description: str
    day: int
    importance: int


class Behavior(BaseModel):
    wearing_mask: bool
    maintaining_social_distance: bool
    self_isolating: bool


class Belief(BaseModel):
    description: str
    sentiment: str


class Beliefs(BaseModel):
    beliefs: list[Belief]


class News(BaseModel):
    headline: str


# %%
# -------------------------------------------------------------------------------------------
# Concurrency Helper for Timeouts
# -------------------------------------------------------------------------------------------
def call_with_timeout(func, timeout=20, *args, **kwargs):
    """
    Runs 'func(*args, **kwargs)' in a separate thread and enforces a 'timeout' in seconds.
    Returns the function's return value if it finishes on time,
    or raises a TimeoutError if it doesn't.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)


# %%
# -------------------------------------------------------------------------------------------
# LLM Integration Functions (with Timeout)
# -------------------------------------------------------------------------------------------
def generate_beliefs(memories: list[Memory], persona: str) -> Beliefs:
    """
    Uses Google Gemini to generate beliefs based on recent memories and the agent's persona,
    enforcing a timeout to avoid hanging.
    """
    memories_json = json.dumps(
        [m.model_dump() for m in memories], ensure_ascii=False, indent=2
    )

    prompt = (
        f"You are an introspective and analytical agent with the following persona: '{persona}'. "
        "Reflect on your recent experiences described in the JSON below and generate a set of beliefs that capture both your emotional reactions and rational assessments of the day's events. "
        "Consider how the events (e.g., interactions, observations related to public health issues like COVID) affect your perception of risk and what lessons you might learn for your future behavior. "
        "Return valid JSON strictly matching this schema:\n\n"
        f"{Beliefs.model_json_schema()}\n\n"
        "Do not include any extraneous keys or commentary.\n\n"
        f"Memories:\n{memories_json}"
        "Please be concise and precise."
    )

    def _call_llm():
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Beliefs,
            },
        )
        print(response)
        return response.parsed

    try:
        beliefs = call_with_timeout(_call_llm, timeout=20)
        return beliefs
    except TimeoutError:
        print("LLM call timed out in generate_beliefs; returning empty beliefs.")
        return Beliefs(beliefs=[])


def generate_behavior(beliefs: list[Belief], persona: str) -> Behavior:
    """
    Uses Google Gemini to generate a daily behavior plan based on current beliefs and persona,
    enforcing a timeout to avoid hanging.
    """
    beliefs_json = json.dumps(
        [b.model_dump() for b in beliefs], ensure_ascii=False, indent=2
    )

    prompt = (
        f"You are an agent with the following persona: '{persona}'. "
        "Based on your current beliefs (provided in JSON format below), please determine your behavior for the day. "
        "Consider the ongoing public health situation (such as the dynamics of COVID-19), the news you have heard, and your personal risk tolerance. "
        "Decide whether to wear a mask, maintain social distance, or self-isolate. "
        "Return valid JSON strictly matching the schema below:\n\n"
        f"{Behavior.model_json_schema()}\n\n"
        "Do not include any extra keys or commentary.\n\n"
        f"Beliefs:\n{beliefs_json}"
        "Please be concise and precise."
    )

    def _call_llm():
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Behavior,
            },
        )
        return response.parsed

    try:
        behavior = call_with_timeout(_call_llm, timeout=20)
        return behavior
    except TimeoutError:
        print("LLM call timed out in generate_behavior; returning default Behavior.")
        return Behavior(
            wearing_mask=False, maintaining_social_distance=False, self_isolating=False
        )


def generate_news(information: str) -> str:
    """
    Uses Google Gemini to generate a news headline based on the current public health context,
    enforcing a timeout to avoid hanging.
    """
    prompt = (
        "You are a seasoned news reporter specializing in public health crises. "
        "Based on the following information about today's epidemic status (e.g., new infections, deaths, changes over previous days), "
        "generate a concise news headline that captures both the factual situation and the public sentiment. "
        "Return your response as valid JSON matching the schema below:\n\n"
        f"{News.model_json_schema()}\n\n"
        "Use only the key 'headline' in your response.\n\n"
        f"Information: '{information}'"
        "Please be concise and precise."
    )

    def _call_llm():
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": News},
        )
        return response.parsed

    try:
        news = call_with_timeout(_call_llm, timeout=20)
        return news.headline
    except TimeoutError:
        print("LLM call timed out in generate_news; returning 'No new updates'.")
        return "No new updates"


# %%
# -------------------------------------------------------------------------------------------
# Agent Class
# -------------------------------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        agent_id,
        age,
        gender,
        home,
        destination,
        destination_type,
        routine,
        network_neighbors,
        persona,
    ):
        self.agent_id = agent_id
        self.age = age
        self.gender = gender
        self.home = home
        self.destination = destination
        self.destination_type = destination_type
        self.routine = routine
        self.network_neighbors = network_neighbors
        self.persona = persona

        self.state = "S"
        self.days_in_state = 0
        self.memories: list[Memory] = []
        self.behavior = Behavior(
            wearing_mask=False, maintaining_social_distance=False, self_isolating=False
        )
        self.beliefs: list[Belief] = []
        self.exposure = 0

        self.locations = [None] * TICKS_PER_DAY
        self.calculate_location()

        self.daily_interactions = []
        self.received_information = []

    def __repr__(self):
        return f"Agent {self.agent_id} ({self.persona})"

    def __str__(self):
        return f"Agent {self.agent_id} ({self.persona})"

    def calculate_location(self):
        # Fill self.locations for each 5-min tick in a day
        for tick in range(TICKS_PER_DAY):
            location_id = self.routine[tick]
            if location_id == self.home:
                self.locations[tick] = (location_id, "Home")
            elif location_id == self.destination:
                self.locations[tick] = (location_id, self.destination_type)
            else:
                # Any other location_id in routine is assumed 'Bus' or intermediate
                self.locations[tick] = (location_id, "Bus")

    def get_location(self, current_tick):
        if self.state == "D":
            # Dead agents aren't moving
            return None, "Dead"
        if self.behavior.self_isolating:
            # If self-isolating, always remain at home
            return self.home, "Home"
        return self.locations[current_tick]

    def interact(self, agent, current_tick):
        # If either agent is dead, skip
        if self.state == "D" or agent.state == "D":
            return

        # Record interaction
        interaction = {
            "agent_id": agent.agent_id,
            "wearing_mask": agent.behavior.wearing_mask,
            "maintaining_social_distance": agent.behavior.maintaining_social_distance,
            "self_isolating": agent.behavior.self_isolating,
            "state": agent.state,
        }
        self.daily_interactions.append(interaction)

        infectious_states = ["E", "IA", "IS", "IC"]

        def exposure_increase(a1, a2, tick):
            exposure_score = 1.0
            if a1.behavior.wearing_mask:
                exposure_score *= BEHAVIORS["wearing_mask"]
            if a1.behavior.maintaining_social_distance:
                exposure_score *= BEHAVIORS["maintaining_social_distance"]
            if a1.behavior.self_isolating:
                exposure_score *= BEHAVIORS["self_isolating"]

            if a2.behavior.wearing_mask:
                exposure_score *= BEHAVIORS["wearing_mask"]
            if a2.behavior.maintaining_social_distance:
                exposure_score *= BEHAVIORS["maintaining_social_distance"]
            if a2.behavior.self_isolating:
                exposure_score *= BEHAVIORS["self_isolating"]

            location_type = a1.get_location(tick)[1]
            exposure_score *= mean_contact_values.get(location_type, 1)
            return exposure_score

        # Transmission
        if self.state in infectious_states and agent.state == "S":
            agent.exposure += exposure_increase(self, agent, current_tick)
        elif agent.state in infectious_states and self.state == "S":
            self.exposure += exposure_increase(agent, self, current_tick)

    def update_state(self):
        """Progress the infection states according to the time spent and random draws."""
        if self.state == "S":
            if self.exposure > 0:
                raw_prob = ALPHA * math.log(1 + self.exposure / 288.0)
                probability = max(0, min(1, raw_prob))
            else:
                probability = 0

            if random.random() < probability:
                self.state = "E"
                self.days_in_state = 0

        elif self.state == "E":
            if self.days_in_state >= TIME_PERIODS["E"]:
                if random.random() < BETA:
                    self.state = "IA"
                else:
                    self.state = "IS"
                self.days_in_state = 0

        elif self.state == "IA":
            if self.days_in_state >= TIME_PERIODS["IA"]:
                if random.random() < GAMMA:
                    self.state = "IS"
                else:
                    self.state = "R"
                self.days_in_state = 0

        elif self.state == "IS":
            if self.days_in_state >= TIME_PERIODS["IS"]:
                if random.random() < THETA:
                    self.state = "IC"
                else:
                    self.state = "R"
                self.days_in_state = 0

        elif self.state == "IC":
            if self.days_in_state >= TIME_PERIODS["IC"]:
                if random.random() < PHI:
                    self.state = "D"
                else:
                    self.state = "R"
                self.days_in_state = 0

        elif self.state == "R":
            if self.days_in_state >= TIME_PERIODS["R"]:
                if random.random() < OMEGA:
                    self.state = "M"
                else:
                    self.state = "S"
                self.days_in_state = 0

        # Reset exposure and increment days_in_state
        self.exposure = 0
        self.days_in_state += 1

    def summarize_observations(self, day):
        """Create a memory summarizing the day's interactions."""
        total_interactions = len(self.daily_interactions)
        mask_wearers = sum(1 for i in self.daily_interactions if i["wearing_mask"])
        social_distancers = sum(
            1 for i in self.daily_interactions if i["maintaining_social_distance"]
        )
        self_isolators = sum(1 for i in self.daily_interactions if i["self_isolating"])

        description = (
            f"On day {day+1}, I interacted with people {total_interactions} times this day. "
            f"{mask_wearers} wore masks, {social_distancers} maintained social distance, "
            f"and {self_isolators} were self-isolating."
        )
        importance = 5
        self.memorize(description, day, importance)

        # Clear daily interactions for the next day
        self.daily_interactions = []

    def receive_information(self, information, day):
        """Add a memory about new information received."""
        description = f"On day {day+1}, I heard that {information}"
        importance = 5
        self.memorize(description, day, importance)

        # Add latest info to received_information
        self.received_information.append((information, day))

    def memorize(self, description, day, importance):
        memory = Memory(description=description, day=day, importance=importance)
        self.memories.append(memory)

    def reflect(self, day):
        # Reflect on the last 10 memories, regardless of day
        recent_memories = self.memories[-10:]

        if not recent_memories:
            self.beliefs = []
            return
        if random.random() < LLM_INFERENCE_PROB:
            beliefs_obj = generate_beliefs(recent_memories, self.persona)
            self.beliefs = beliefs_obj.beliefs

    def plan(self):
        # Use the last 3 beliefs to generate a behavior plan
        recent_beliefs = self.beliefs[:3]

        if not self.beliefs:
            return
        if random.random() < LLM_INFERENCE_PROB:
            self.behavior = generate_behavior(recent_beliefs, self.persona)


# %%
# -------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------
def get_agents_db():
    """Read the 'agents' table from agents.db into a list of dicts."""
    conn = sqlite3.connect("agents.db")
    pd_agents = pd.read_sql_query("SELECT * FROM agents", conn)
    conn.close()
    return pd_agents.to_dict(orient="records")


def get_network_neighbors_db(agent_id):
    """Retrieve direct neighbors of a given agent_id from the 'network' table."""
    conn = sqlite3.connect("agents.db")
    cur = conn.cursor()
    cur.execute("SELECT agent2 FROM network WHERE agent1 = ?", (agent_id,))
    neighbors = [row[0] for row in cur.fetchall()]
    conn.close()
    return neighbors


def get_overall_stats(agents: list[Agent]):
    """Aggregate and return simulation-wide counts of each disease state."""
    num_susceptible = sum(1 for a in agents if a.state == "S")
    num_exposed = sum(1 for a in agents if a.state == "E")
    num_asymptomatic = sum(1 for a in agents if a.state == "IA")
    num_seriously_infected = sum(1 for a in agents if a.state == "IS")
    num_critically_infected = sum(1 for a in agents if a.state == "IC")
    num_recovered = sum(1 for a in agents if a.state == "R")
    num_immune = sum(1 for a in agents if a.state == "M")
    num_dead = sum(1 for a in agents if a.state == "D")
    return {
        "Susceptible": num_susceptible,
        "Exposed": num_exposed,
        "Asymptomatic": num_asymptomatic,
        "Seriously Infected": num_seriously_infected,
        "Critically Infected": num_critically_infected,
        "Recovered": num_recovered,
        "Immune": num_immune,
        "Dead": num_dead,
    }


# %%
# -------------------------------------------------------------------------------------------
# Constructing the Network
# -------------------------------------------------------------------------------------------
conn = sqlite3.connect("agents.db")
pd_agents = pd.read_sql_query("SELECT * FROM agents", conn)
conn.close()
agents_raw = pd_agents.to_dict(orient="records")

G = nx.Graph()

# Add nodes for each agent
for agent in agents_raw:
    G.add_node(agent["agent_id"])

# Group by home and workplace to add strong ties
home_groups = {}
workplace_groups = {}

for agent in agents_raw:
    home_id = agent["home"]
    if home_id not in home_groups:
        home_groups[home_id] = []
    home_groups[home_id].append(agent["agent_id"])

    # Only add workplace if destination_type is "Workplace"
    if agent["destination_type"] == "Workplace":
        workplace_id = agent["destination"]
        if workplace_id not in workplace_groups:
            workplace_groups[workplace_id] = []
        workplace_groups[workplace_id].append(agent["agent_id"])

# Create strong ties (home)
for home_id, members in home_groups.items():
    if len(members) > 1:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                G.add_edge(members[i], members[j])

# Create strong ties (workplace)
for workplace_id, members in workplace_groups.items():
    if len(members) > 1:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                G.add_edge(members[i], members[j])

# Add random weak ties (friends)
num_agents = len(agents_raw)
agent_ids = [a["agent_id"] for a in agents_raw]
random_friend_connections = 3000
for _ in range(random_friend_connections):
    agent1 = random.choice(agent_ids)
    agent2 = random.choice(agent_ids)
    if agent1 != agent2 and not G.has_edge(agent1, agent2):
        G.add_edge(agent1, agent2)

# Write the network to DB
conn = sqlite3.connect("agents.db")
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS network (agent1 INT, agent2 INT)")
cur.execute("DELETE FROM network")  # Clear previous data
for edge in G.edges():
    cur.execute(
        "INSERT INTO network (agent1, agent2) VALUES (?, ?)", (edge[0], edge[1])
    )
    cur.execute(
        "INSERT INTO network (agent1, agent2) VALUES (?, ?)", (edge[1], edge[0])
    )
conn.commit()
conn.close()

# %%
# -------------------------------------------------------------------------------------------
# Load Agents into Simulation
# -------------------------------------------------------------------------------------------
agents_db = get_agents_db()
agents = []
for agent_db in agents_db:
    routine = agent_db["routine"].split(",")
    neighbors = get_network_neighbors_db(agent_db["agent_id"])
    persona = (
        f"He/She is a {agent_db['age']} year old {agent_db['gender']} who goes to {agent_db['destination_type']} everyday. "
        f"He/She has personality: Openness={agent_db['personality_openness']}, Conscientiousness={agent_db['personality_conscientiousness']}, "
        f"Extraversion={agent_db['personality_extraversion']}, Agreeableness={agent_db['personality_agreeableness']}, Neuroticism={agent_db['personality_neuroticism']}. "
        f"Demographic: Education={agent_db['demographic_education']}, Income={agent_db['demographic_income_level']}. "
        f"Health: Immune Strength={agent_db['health_immune_system_strength']}, Pre-existing Conditions={agent_db['health_pre_existing_conditions']}. "
        f"Psychographic: RiskAttitude={agent_db['psychographic_risk_attitude']}, BeliefSystem={agent_db['psychographic_belief_system']}, "
        f"InfoSensitivity={agent_db['psychographic_information_sensitivity']}, FearLevel={agent_db['psychographic_fear_level']}. "
        f"Behavior: Routine={agent_db['behavior_routine']}, Compliance={agent_db['behavior_compliance_level']}."
    )
    agent = Agent(
        agent_id=agent_db["agent_id"],
        age=agent_db["age"],
        gender=agent_db["gender"],
        home=agent_db["home"],
        destination=agent_db["destination"],
        destination_type=agent_db["destination_type"],
        routine=routine,
        network_neighbors=neighbors,
        persona=persona,
    )
    agents.append(agent)

# Build a lookup dict for agent_id -> Agent object
agent_lookup = {a.agent_id: a for a in agents}

# Initialize some agents as infected
initial_infected = random.sample(agents, k=10)
for agent in initial_infected:
    agent.state = "E"
    agent.days_in_state = 0

stats_records = []

# %%
# -------------------------------------------------------------------------------------------
# Main Simulation Loop
# -------------------------------------------------------------------------------------------
from tqdm import tqdm  # Added import for tqdm

# -------------------------------------------------------------------------------------------
# Main Simulation Loop (with tqdm logging)
# -------------------------------------------------------------------------------------------
for day in tqdm(range(NUM_DAYS), desc="Simulation Days"):
    # Each tick, group agents by location
    for tick in tqdm(range(TICKS_PER_DAY), desc="Ticks", leave=False):
        locations_dict = defaultdict(list)

        # Group agents by current location
        for agent in agents:
            if agent.state == "D":
                continue
            loc_id, loc_type = agent.get_location(tick)
            if loc_id is not None and loc_type != "Dead":
                locations_dict[(loc_id, loc_type)].append(agent)

        # Interactions among agents in the same location
        for (loc_id, loc_type), agents_at_loc in locations_dict.items():
            n_agents_loc = len(agents_at_loc)
            if n_agents_loc > 1:
                for i in range(n_agents_loc):
                    for j in range(i + 1, n_agents_loc):
                        agent_i = agents_at_loc[i]
                        agent_j = agents_at_loc[j]
                        agent_i.interact(agent_j, tick)
                        agent_j.interact(agent_i, tick)

    # End of day summarization
    for agent in agents:
        agent.summarize_observations(day)

    # Compute stats
    stats = get_overall_stats(agents)
    stats_records.append(stats)
    num_infected = (
        stats["Exposed"]
        + stats["Asymptomatic"]
        + stats["Seriously Infected"]
        + stats["Critically Infected"]
    )

    # Generate & spread news if we have at least two days of data
    if len(stats_records) >= 2:
        prev_stats = stats_records[-2]
        prev_infected = (
            prev_stats["Exposed"]
            + prev_stats["Asymptomatic"]
            + prev_stats["Seriously Infected"]
            + prev_stats["Critically Infected"]
        )

        # Run the LLM to produce a headline with a probability
        if random.random() < LLM_INFERENCE_PROB:
            headline = generate_news(
                f"Today, new infections: {num_infected}, yesterday: {prev_infected}, "
                f"deaths: {stats['Dead']} (changed by {stats['Dead'] - prev_stats['Dead']})."
            )
        else:
            headline = "No new updates"

        # Spread news to a random 10% of agents
        num_agents_to_receive_news = int(0.1 * len(agents))
        if num_agents_to_receive_news > 0:
            agents_receiving_news = random.sample(agents, num_agents_to_receive_news)
            for a in agents_receiving_news:
                a.receive_information(headline, day)

    # Agents pass info to neighbors
    for agent in agents:
        # new list to keep unexpired info
        new_info_list = []
        for information, info_day in agent.received_information:
            # If not older than 7 days, we spread it
            if (day - info_day) <= 7:
                pass_probability = INFORMATION_PROB
                for neighbor_id in agent.network_neighbors:
                    if random.random() < pass_probability:
                        neighbor_agent = agent_lookup.get(neighbor_id, None)
                        if neighbor_agent:
                            neighbor_agent.receive_information(information, day)

                # Keep this info for future days
                new_info_list.append((information, info_day))

        # Update agent's info
        agent.received_information = new_info_list

    # Update infection states
    for agent in agents:
        agent.update_state()

    # Reflection & Planning using LLM
    for agent in agents:
        # print(f"Agent {agent.agent_id} is reflecting...")
        agent.reflect(day)
        # print(f"Agent {agent.agent_id} is planning...")
        agent.plan()

    # Print daily stats
    print(
        f"Day {day+1} stats: "
        f"S: {stats['Susceptible']}, E: {stats['Exposed']}, IA: {stats['Asymptomatic']}, "
        f"IS: {stats['Seriously Infected']}, IC: {stats['Critically Infected']}, "
        f"R: {stats['Recovered']}, M: {stats['Immune']}, D: {stats['Dead']}"
    )

# %%
# -------------------------------------------------------------------------------------------
# Final Visualization
# -------------------------------------------------------------------------------------------
G_final = nx.Graph()
for a in agents:
    G_final.add_node(a.agent_id, state=a.state)

for a in agents:
    for neighbor_id in a.network_neighbors:
        G_final.add_edge(a.agent_id, neighbor_id)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_final, seed=42)
node_colors = []
for a in agents:
    if a.state == "S":
        node_colors.append("green")
    elif a.state == "E":
        node_colors.append("orange")
    elif a.state == "IA":
        node_colors.append("pink")
    elif a.state == "IS":
        node_colors.append("red")
    elif a.state == "IC":
        node_colors.append("darkred")
    elif a.state == "R":
        node_colors.append("blue")
    elif a.state == "M":
        node_colors.append("yellow")
    elif a.state == "D":
        node_colors.append("black")

nx.draw_networkx_nodes(G_final, pos, node_size=20, node_color=node_colors)
nx.draw_networkx_edges(G_final, pos, alpha=0.1)
plt.title("Agent Network Visualization")
plt.axis("off")
plt.show()

# %%
