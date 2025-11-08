import uuid
# this defines the state objects that we will use

# cost can be a static cost of a unit, or a dynamic cost calculated at each step
class Cost:
    build_power: float
    metal: float
    energy: float

    def __init__(self, bp, m, e):
        self.build_power = bp
        self.metal = m
        self.enery = e

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
        self.metal_per_second = ms
        self.energy_per_second = es

    def __add__(self, other):
        if other is None:
            return self
        else:
            return Income(
                self.metal_per_second + other.metal_per_second,
                self.energy_per_second + other.energy_per_second
            )

class Position:
    x: int
    y: int

class Production:
    id: uuid.UUID
    type: str
    cost: Cost # this should just be the total cost
    completion: float
    bp_applied: float

    def __init__(self, cost, type):
        self.cost = cost
        self.completion = 0
        self.bp_applied = 0
        self.id = uuid.uuid4()
        self.type = type

    # return the maximum possible expenditure on this product this time step based on applied build power
    def step_max_cost(self):
        # we cannot apply more than the build power required to complete
        bp_app = min(self.bp_applied, 1 - self.cost.build_power)
        metal = self.cost.metal * bp_app/self.cost.build_power
        energy = self.cost.energy * bp_app/self.cost.build_power
        return Cost(bp_app, metal, energy)

    # update complete based on actual spent build power
    def update_completion(self, cost: Cost):
        self.completion += cost.build_power/self.cost.build_power


class Resource:
    type: str
    move_speed: int
    build_power: int
    position: Position
    can_build: list[str]
    income: Income | None
    current_action: Production

    def __init__(self, type, id):
        self.id = id
        self.type = type
    


class BARState:
    timestep: int
    resources: list[Resource]
    productions: list[Production]
    income: Income
    total_metal: float
    total_energy: float
    max_enery: float
    energy_convert_percent: float = 0.75

    def __init__(self):
        self.timestep = 0
        self.resources = []
        self.productions = []
        self.income = Income(0, 0)

    def step_expenditure_and_production(self) -> Cost:
        all_costs: list[Cost] = [p.step_max_cost() for p in self.productions]
        max_cost: Cost = sum(all_costs)
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
                complete.append[p]
                self.resources.append(Resource(p.type, p.id))
        self.productions.remove(p)
        return sum(all_costs)

    def calc_energy_conversion(self):
        return 0

    def step_income(self, total_cost: Cost):
        # first, if there is any energy available, we can spend it with energy converters
        # in game this always happens after build costs are taken into account
        metal_prod = 0
        energy_prod = 0
        metal_prod += self.calc_energy_conversion()
        total_incomes: Income = sum([r.income for r in self.resources if r.income is not None])
        total_incomes.metal_per_second += metal_prod
        total_incomes.energy_per_second += energy_prod
        self.total_energy += total_incomes.energy_per_second - total_cost.energy
        self.total_metal += total_incomes.metal_per_second - total_cost.metal
