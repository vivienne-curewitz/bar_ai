import uuid
import time
import math
from data import load_cortex_data, build_power
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import threading
from multiprocessing import Process, Queue
from functools import reduce
# this defines the state objects that we will use

units, structures = load_cortex_data()
u2 = units.copy()
s2 = structures.copy()
u2["kind"] = "unit"
s2["kind"] = "structure"
u2 = u2.rename(columns={"Unit": "Name"})
s2 = s2.rename(columns={"Structure": "Name"})

cortex_resources = pd.concat([u2, s2], ignore_index=True)

# cost can be a static cost of a unit, or a dynamic cost calculated at each step
class Cost:
    build_power: float
    metal: float
    energy: float

    def __init__(self, bp, m, e):
        self.build_power = bp
        self.metal = m
        self.energy = e

    def __add__(self, other):
        if not isinstance(other, Cost):
            raise NotImplementedError
        return Cost(self.build_power + other.build_power, 
                    self.metal + other.metal,
                    self.energy + other.energy)
    def NewCost(action):
        p_row = cortex_resources.loc[cortex_resources["Name"] == action]
        return Cost(p_row['BC'].iloc[0], p_row['MC'].iloc[0], p_row['EC'].iloc[0])

    # given a dynamic cost X, scale it by ratio to be (almost always) lower
    def scale(self, ratio):
        if ratio >= 1:
            raise ValueError("Cost scale ratio should be less than 1")
        self.build_power *= ratio
        self.metal *= ratio
        self.energy *= ratio

class Income:
    metal_per_second: float
    energy_per_second: float

    def __init__(self, ms, es):
        self.metal_per_second = ms if not math.isnan(ms) else 0
        self.energy_per_second = es if not math.isnan(es) else 0

    def __add__(self, other):
        if other is None:
            return self
        else:
            return Income(
                self.metal_per_second + other.metal_per_second,
                self.energy_per_second + other.energy_per_second
            )

    def __str__(self):
        return f"{self.metal_per_second} m/s -- {self.energy_per_second} e/s"

class Position:
    x: int
    y: int

class Resource:
    type: str
    category: str # will be structure or unit
    move_speed: int
    build_power: int
    can_build: list[str]
    income: Income | None
    cost: Cost | None
    tier: int

    def __init__(self, type, id, df_row, category):
        self.id = id
        self.type = type
        self.category = category
        self.current_action = None
        self.can_build = []
        self.income = Income(df_row['MP'].iloc[0], df_row['EP'].iloc[0])
        self.tier = df_row["Tier"].iloc[0]
        if self.type in build_power.keys():
            self.build_power = build_power[self.type]
        else:
            self.build_power = 0
        if self.category == "unit": # and type != "Cortex Commander":
            if "Construction" in self.type:
                self.can_build = structures.loc[(structures['Tier'] == self.tier) & (structures["Requires"].isna())]['Structure'].to_list()
                if self.tier == 2:
                    self.can_build.append("Experimental Gantry")
                elif self.tier == 1:
                    matches = structures[structures["Requires"] == self.type]
                    for structure in matches["Structure"]:
                        self.can_build.append(structure)
            elif "Commander" in self.type:
                self.can_build = structures.loc[(structures['Tier'] == 1) & (structures["Requires"].isna())]['Structure'].to_list()
        elif self.category == "structure":
            self.can_build = units.loc[units['Factory'] == self.type]['Unit'].to_list()

    def __str__(self):
        return self.type

    def NewResource(type):
        df_row = cortex_resources.loc[cortex_resources["Name"] == type]
        return Resource(type, id, df_row, category=df_row["kind"].iloc[0]) 



class BARState:
    resources: list[Resource]
    income: Income # not being used
    energy_convert_percent: float = 0.75
    mex_available = 3 # this is a weird one, the player is always limited to some number of mexes they can build, but everything else is unlimited (besides geos)
    incoming_edges = []
    outgoing_edges = []
    heuristic = 10000000000000000000000000000 # just really large

    def __init__(self, resources = []):
        self.resources = resources
        self.income = Income(0, 0)

    def utility(self) -> float:
        if self.income.metal_per_second == 0:
            return 0
        if self.income.energy_per_second == 0:
            return 0
        total_bp = sum([r.build_power for r in self.resources])
        # utility is just the sum of all resources and incomes
        return total_bp + self.income.energy_per_second + self.income.metal_per_second 

    def calc_bp_applied(self, cost: Cost) -> Income:
        all_incomes = [r.income for r in self.resources if r.income is not None and not "Converter" in r.type]
        total_incomes = 0
        if len(all_incomes) >= 1:
            total_incomes: Income = sum(all_incomes[1:], all_incomes[0])
        if total_incomes == 0:
            total_incomes = Income(0, 0)
        total_bp = sum([r.build_power for r in self.resources])
        max_eps = min(cost.energy *(total_bp/cost.build_power), cost.energy)
        max_mps = min(cost.metal * (total_bp/cost.build_power), cost.energy)
        # this part came from chatty
        x = 1.0
        Se = total_incomes.energy_per_second
        Sm = total_incomes.metal_per_second
        C = sum([r.income.metal_per_second for r in self.resources if "Converter" in r.type])
        R = 70
        if C > 0 and Se > max_eps and Sm < max_mps:
            # print("Doing coverter loop")
            for _ in range(6):
                metal_demand = x * max_mps
                energy_demand = x * max_eps

                surplus = max(0, Se - energy_demand)
                conv_mps = min(C, surplus / R)

                metal_avail = Sm + conv_mps
                energy_avail = Se

                x_new = min(metal_avail / max_mps, energy_avail / max_eps, 1.0)

                if abs(x_new - x) < 1e-6:
                    break
                x = x_new
        bp_applied = x * total_bp
        return bp_applied
        
    def expand_state(self):
        if self.mex_available > 0:
            possible_action = set([action for r in self.resources for action in r.can_build])
        else:
            possible_action = set([action for r in self.resources for action in r.can_build if "Metal" not in action])
        new_states = []
        new_edges = []
        for action in possible_action:
            # new states defined here
            cost = Cost.NewCost(action)
            bp_applied = self.calc_bp_applied(cost)
            gen_resource = Resource.NewResource(type=action)
            new_res = [res for res in self.resources]
            new_res.append(gen_resource)
            state_nasta = BARState(resources=new_res)
            edge = StateEdge(self, state_nasta, cost.build_power, bp_applied)
            self.outgoing_edges.append(edge)
            state_nasta.incoming_edges.append(edge) # who knows if I will ever use this
            new_states.append(state_nasta)
            new_edges.append(edge)
        return new_states, new_edges

    def resource_count(self) -> dict:
        my_r_count = {}
        for r in self.resources:
            if r.type in my_r_count.keys():
                my_r_count[r.type] += 1
            else:
                my_r_count[r.type] = 1
        return my_r_count

    def compare(self, other):
        my_count = self.resource_count()
        their_count = other.resource_count()
        return my_count == their_count 

    # return float between 0 and 1 which is the ratio of items in self to items in other
    def isSubset(self, other:dict) -> float:
        me = self.resource_count()
        total_other = 0
        my_hits = 0
        for u, c in other:
            total_other += c
            if u in me.keys():
                my_hits += min(me[u], other[u])
        return my_hits/total_other
    
    def goalDelta(self, goal: dict) -> dict:
        me = self.resource_count()
        delta = {}
        for u, c in goal.items():
            delta[u] = goal[u] - me[u] if u in me.keys() else goal[u]
        return delta

    def delta_distance(self, delta: dict):
        total_time = 0
        for u, c in delta.items():
            u_cost = Cost.NewCost(u)
            bp_app = self.calc_bp_applied(u_cost)
            total_time += u_cost.build_power/bp_app * c
        return total_time

# goal is a dict of the form {Unit: str, count: int}
def generate_goal_state(goal: dict):
    # generate all resources once:
    all_resources = [Resource.NewResource(name) for name in cortex_resources["Name"].to_list()]
    required_resources = []
    for u in goal.keys():
        tres = Resource.NewResource(u)
        reqs = requirement_chain(tres, all_resources)
        for req in flatten_tuple(reqs):
            required_resources.append(req)
    state_dict = {}
    for k in set(required_resources):
        state_dict[k.type] = 1
    for u, count in goal.items():
        state_dict[u] = count
    return state_dict
     
def flatten_tuple(nested_tuple):
    def reducer(acc, val):
        if isinstance(val, tuple):
            return acc + flatten_tuple(val)
        else:
            return acc + (val,)

    return reduce(reducer, nested_tuple, ())

def requirement_chain(unit: Resource, all_resources: list[Resource]) -> tuple[str]:
    if unit.category == "structure" and unit.tier == 1 and len(unit.can_build) > 0:
        return unit
    potentials = []
    for r in all_resources:
        if unit.type in r.can_build and (r.type not in unit.can_build or r.tier < unit.tier):
            return requirement_chain(r, all_resources), unit
        elif unit.type in r.can_build:
            potentials.append(r)
    if len(potentials):
        return requirement_chain(potentials[0], all_resources), unit
    return unit    
    
class StateEdge:
    prev_state: BARState
    post_state: BARState
    bp_delta: int
    bp_applied: int

    def __init__(self, pre: BARState, post: BARState, bp_delta: int, bp_applied: int):
        self.prev_state = pre
        self.post_state = post
        self.bp_applied = bp_applied
        self.bp_delta = bp_delta

    # never return less than 1
    @property
    def distance(self) -> float:
        return max(self.bp_delta/self.bp_applied, 1.0)

class BARStateGraph:
    states = []
    edges = []
    count = 0
    start_time = 0
    bench_info = []
    best_score = 0000

    def __init__(self, states=[], edges=[]):
        self.states=states
        self.edges=edges

    def expand_graph(self, state_index, heuristic):
        if len(self.states) <= state_index:
            print(f"Trying to work on state outside of array bounds")
        state: BARState = self.states[state_index]
        new_states, new_edges = state.expand_state()
        # for i, ns in enumerate(new_states):
        #     to_add = True
        #     for es in self.states:
        #         if ns.compare(es):
        #             new_edges[i].post_state = es
        #             to_add = False
        #             break
        #     if to_add:
        #         ns.heuristic = heuristic(state)
        #         self.states.append(ns)
        #     self.edges.append(new_edges[i])
        for i, ns in enumerate(new_states):
            ns.heuristic = heuristic(state)
            self.states.append(ns)
            self.edges.append(new_edges[i])

    def heuristic_factory(self, goal, utility_ratio = 0, distance_ratio=1):
        def heuristic(state: BARState):
            delta = state.goalDelta(goal)
            return distance_ratio * state.delta_distance(delta) - utility_ratio * state.utility() - len(state.resources)
        return heuristic
    """
    Debug and testing functions below
    """

    def info(self, running_thread: threading.Thread, queue: Queue):
        while running_thread.is_alive():
            print(f"States Expaneded: {self.count}\nTotal States: {len(self.states)}\nRuntime: {time.perf_counter() - self.start_time}")
            snap = {
                "time": time.perf_counter() - self.start_time,
                "SE": self.count,
                "TS": math.log(len(self.states)),
                "BS": self.best_score
            }
            queue.put(snap)
            time.sleep(0.05)

    def graph_process(queue: Queue):
        bench_info = []
        # Initialize the plot
        plt.ion()  # interactive mode on
        fig, ax = plt.subplots(figsize=(10, 5))
        line_se, = ax.plot([], [], label="States Expanded")
        line_ts, = ax.plot([], [], label="Total States")
        line_bs, = ax.plot([], [], label="Best Score")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
        ax.set_title("Expansion Progress")
        ax.legend()
        ax.grid(True)
        while True:
            snap = queue.get() 
            if snap == "done":
                return
            bench_info.append(snap)
            # Convert to DataFrame for plotting
            df = pd.DataFrame(bench_info)
            # Update line data
            line_se.set_data(df["time"], df["SE"])
            line_ts.set_data(df["time"], df["TS"])
            line_bs.set_data(df["time"], df["BS"])
            # Rescale axes
            ax.relim()
            ax.autoscale_view()
            # Redraw
            fig.canvas.draw()
            fig.canvas.flush_events()

    def test_goal_find(self, i=1000, goal={}):
        heuristic = self.heuristic_factory(goal, 0, 1)
        self.start_time = time.perf_counter()
        def search_loop():
            smallest = 0
            last_smallest = -1
            while self.count < i:
                self.expand_graph(smallest, heuristic)
                hs = 10000000000000000000000
                for j, ns in enumerate(self.states):
                    if ns.heuristic < hs:
                        smallest = j 
                        hs = ns.heuristic
                self.best_score = hs
                print(f"Best State Score: {self.states[smallest].heuristic} in {len(self.states)} states")
                if smallest == last_smallest:
                    print(f"Error - found same smallest value\n{self.states[j].resource_count()}")
                    return
                else:
                    last_smallest = smallest 
                self.count += 1
                if hs <= 0:
                    print(f"Found Solution: {self.states[smallest].resource_count()}")
                    return
        compute_thread = threading.Thread(target=search_loop)
        compute_thread.start()
        queue = Queue()
        graph_p = Process(target=BARStateGraph.graph_process, args=[queue])
        graph_p.start()        
        self.info(compute_thread, queue)
        queue.put("done")



    def test_loop(self, i=1000):
        heuristic = self.heuristic_factory({}, 0, 1)
        self.start_time = time.perf_counter()
        def expand_loop():
            while self.count < i:
                self.expand_graph(self.count, heuristic)
                self.count += 1
                time.sleep
        compute_thread = threading.Thread(target=expand_loop)
        compute_thread.start()
        queue = Queue()
        graph_p = Process(target=BARStateGraph.graph_process, args=[queue])
        graph_p.start()        
        self.info(compute_thread, queue)
        queue.put("done")

def quick_test():
    type = "Construction Bot"
    df_row = units.loc[units["Unit"] == type]
    cat = "unit"
    cb = Resource(type, "adfasdfasdf", df_row, cat)
    print(f"CB can build: {cb.can_build}")
    type = "Advanced Construction Bot"
    df_row = units.loc[units["Unit"] == type]
    cat = "unit"
    cb = Resource(type, "adfasdfasdf", df_row, cat)
    print(f"ACB can build: {cb.can_build}")   
    type = "Bot Lab"
    df_row = structures.loc[structures["Structure"] == type]
    cat = "structure"
    cb = Resource(type, "adfasdfasdf", df_row, cat)
    print(f"Bot Lab can build: {cb.can_build}")   

def test_req_chain():
    all_resources = [Resource.NewResource(name) for name in cortex_resources["Name"].to_list()]
    reqs = flatten_tuple(requirement_chain(Resource.NewResource("Tzar"), all_resources))
    for req in reqs:
        print(req.type)
    goal = {"Tzar": 4}
    print(generate_goal_state(goal))


def step_state():
    com = Resource.NewResource("Cortex Commander")
    base_state = BARState([com])
    graph = BARStateGraph(states=[base_state])
    graph.test_goal_find(i=1000, goal={"Tzar": 4})

if __name__ == "__main__":
    step_state()
    # test_req_chain()