import uuid
import math
from data import load_cortex_data, build_power
from random import randint
# this defines the state objects that we will use

units, structures = load_cortex_data()

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

class Production:
    id: uuid.UUID
    type: str
    cost: Cost # this should just be the total cost
    completion: float
    bp_applied: float
    owner: object | None

    def __init__(self, cost, type):
        self.cost = cost
        self.completion = 0
        self.bp_applied = 0
        self.id = uuid.uuid4()
        self.type = type
        self.owner = None

    def __str__(self):
        return f"{self.type} -- {self.completion*100}%"

    # return the maximum possible expenditure on this product this time step based on applied build power
    def step_max_cost(self):
        # we cannot apply more than the build power required to complete
        bp_app = min(self.bp_applied, (1 - self.completion)*self.cost.build_power)
        metal = self.cost.metal * bp_app/self.cost.build_power
        energy = self.cost.energy * bp_app/self.cost.build_power
        return Cost(bp_app, metal, energy)

    # update complete based on actual spent build power
    def update_completion(self, cost: Cost):
        self.completion += cost.build_power/self.cost.build_power


class Resource:
    type: str
    category: str # will be structure or unit
    move_speed: int
    build_power: int
    position: Position
    can_build: list[str]
    income: Income | None
    current_action: Production

    def __init__(self, type, id, df_row, category):
        self.id = id
        self.type = type
        self.category = category
        self.current_action = None
        self.can_build = []
        self.income = Income(df_row['MP'].iloc[0], df_row['EP'].iloc[0])
        if self.type in build_power.keys():
            self.build_power = build_power[self.type]
        else:
            self.build_power = 0
        if self.category == "unit": # and type != "Cortex Commander":
            if "Advanced Construction" in self.type:
                tier = 2.0
            elif "Construction" in self.type or "Commander" in self.type:
                tier = 1.0
            self.can_build = structures.loc[structures['Tier'] == tier]['Structure'].to_list()
            if "Advanced" in self.type:
                self.can_build.append("Experimental Gantry")
        elif self.category == "structure":
            self.can_build = units.loc[units['Factory'] == self.type]['Unit'].to_list()

    def __str__(self):
        return self.type

    def NewResource(type, category, id=uuid.uuid4()):
        df_row = None
        if category == "unit":
            df_row = units.loc[units["Unit"] == type]
        else:
            df_row = structures.loc[structures["Structure"] == type]
        return Resource(type, id, df_row, category) 


class BARState:
    timestep: int
    resources: list[Resource]
    productions: list[Production]
    income: Income # not being used
    total_metal: float = 1000
    total_energy: float = 1000
    max_energy: float = 1500
    max_metal: float = 1500
    energy_convert_percent: float = 0.75
    mex_available = 3

    def __init__(self):
        self.timestep = 0
        self.resources = []
        self.productions = []
        self.income = Income(0, 0)

    def utility(self):
        if self.income.metal_per_second == 0 and self.total_metal == 0:
            return 0
        if self.income.energy_per_second == 0 and self.total_energy == 0:
            return 0
        total_bp = sum([r.build_power for r in self.resources])
        # utility is just the sum of all resources and incomes
        return total_bp + self.income.energy_per_second + self.income.metal_per_second + self.total_energy/10 + self.total_metal/10

    def loop(self, max_steps=1000):
        step = 0
        while step < max_steps:
            total_cost = self.step_expenditure_and_production()
            self.step_income(total_cost)
            self.select_possible_actions()
            res_list = [f"{r}" for r in self.resources]
            prod_list = [f"{p}" for p in self.productions]
            print(f"Timestep: {self.timestep} Utilty: {self.utility()} Income: {self.income} Total: {self.total_metal:.0f} m {self.total_energy:.0f}") # e\nResources: {res_list}\nProductions: {prod_list}")
            # next loop
            self.timestep += 1
            step += 1

    def step_expenditure_and_production(self) -> Cost:
        all_costs: list[Cost] = [p.step_max_cost() for p in self.productions]
        if len(all_costs) == 0:
            return Cost(0, 0, 0)
        max_cost: Cost = sum(all_costs[1:], all_costs[0])        
        # we cannot meet max cost, so spend will be lower
        if max_cost.metal > self.total_metal or max_cost.energy > self.total_energy:
            m_spend_ratio = self.total_metal/max_cost.metal
            e_spend_ratio = self.total_energy/max_cost.energy
            spend_ratio = min(m_spend_ratio, e_spend_ratio)
            for cost in all_costs:
                cost.scale(spend_ratio)
        # this only works because lists are ordered
        complete = []
        for i, p in enumerate(self.productions):
            p.update_completion(all_costs[i])
            if p.completion >= 1:
                complete.append(p)
                self.resources.append(Resource.NewResource(type=p.type, category="unit" if p.owner.type == "structure" else "structure"))
        for p in complete:
            self.productions.remove(p)
            p.owner.current_action = None
        return sum(all_costs[1:], all_costs[0])

    def calc_energy_conversion(self):
        return 0

    def step_income(self, total_cost: Cost):
        # first, if there is any energy available, we can spend it with energy converters
        # in game this always happens after build costs are taken into account
        metal_prod = 0
        energy_prod = 0
        metal_prod += self.calc_energy_conversion()
        all_incomes = [r.income for r in self.resources if r.income is not None]
        total_incomes = 0
        if len(all_incomes) >= 1:
            total_incomes: Income = sum(all_incomes[1:], all_incomes[0])
        if total_incomes == 0:
            total_incomes = Income(0, 0)
        self.income = total_incomes
        total_incomes.metal_per_second += metal_prod
        total_incomes.energy_per_second += energy_prod
        self.total_energy = min(self.total_energy + total_incomes.energy_per_second - total_cost.energy, self.max_energy)
        self.total_metal = min(self.total_metal + total_incomes.metal_per_second - total_cost.metal, self.max_metal)

    def select_possible_actions(self):
        current_actions = [r.current_action for r in self.resources if r.current_action is not None]
        for r in self.resources:
            if len(r.can_build) > 0 and r.current_action is None:
                # each option in can build goes to a new state -- I guess we sim each state and aggresively prune based on imporved utility
                # lets prune ahead of time to remove states
                # for now hard code to wind to test
                ind = randint(0,1)
                next_b = [b for b in r.can_build if "Wind" in b or "Metal" in b]
                ns = next_b[ind]
                if "Metal" in ns and self.mex_available == 0:
                    ns = next_b[int(not ind)]
                if "Metal" in ns:
                    self.mex_available -= 1
                p_row = structures.loc[structures["Structure"] == ns]
                cost = Cost(p_row['BC'].iloc[0], p_row['MC'].iloc[0], p_row['EC'].iloc[0])
                new_prod = Production(cost=cost, type=ns)
                new_prod.owner = r
                new_prod.bp_applied = r.build_power
                r.current_action = new_prod
                self.productions.append(new_prod)

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

def step_state():
    com = Resource.NewResource("Cortex Commander", "unit")
    base_state = BARState()
    base_state.resources.append(com) # initial beginning of game state
    base_state.loop(300)

if __name__ == "__main__":
    step_state()