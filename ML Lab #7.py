
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from itertools import product
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

model = BayesianNetwork([
    ('Attendance', 'StudyTime'),
    ('SocialLife', 'StudyTime'),
    ('Attendance', 'Performance'),
    ('StudyTime', 'Performance'),
    ('Health', 'Performance')
])

attendance_cpd = TabularCPD(
    variable='Attendance',
    variable_card=2,
    values=[[0.35], [0.65]],
    state_names={'Attendance': ['Low', 'High']}
)

sociallife_cpd = TabularCPD(
    variable='SocialLife',
    variable_card=3,
    values=[[0.25], [0.50], [0.25]],
    state_names={'SocialLife': ['Low', 'Balanced', 'High']}
)

health_cpd = TabularCPD(
    variable='Health',
    variable_card=2,
    values=[[0.30], [0.70]],
    state_names={'Health': ['Poor', 'Good']}
)

studytime_cpd = TabularCPD(
    variable='StudyTime',
    variable_card=3,
    values=[
        [0.65, 0.40, 0.30, 0.25, 0.15, 0.10],
        [0.30, 0.45, 0.50, 0.50, 0.50, 0.40],
        [0.05, 0.15, 0.20, 0.25, 0.35, 0.50]
    ],
    evidence=['Attendance', 'SocialLife'],
    evidence_card=[2, 3],
    state_names={
        'StudyTime': ['Low', 'Medium', 'High'],
        'Attendance': ['Low', 'High'],
        'SocialLife': ['Low', 'Balanced', 'High']
    }
)

performance_cpd = TabularCPD(
    variable='Performance',
    variable_card=3,
    values=[
        [0.75, 0.60, 0.50, 0.35, 0.20, 0.10,
         0.55, 0.40, 0.25, 0.15, 0.08, 0.03],
        [0.22, 0.32, 0.35, 0.35, 0.30, 0.25,
         0.40, 0.45, 0.45, 0.40, 0.27, 0.17],
        [0.03, 0.08, 0.15, 0.30, 0.50, 0.65,
         0.05, 0.15, 0.30, 0.45, 0.65, 0.80]
    ],
    evidence=['Attendance', 'StudyTime', 'Health'],
    evidence_card=[2, 3, 2],
    state_names={
        'Performance': ['Poor', 'Average', 'Excellent'],
        'Attendance': ['Low', 'High'],
        'StudyTime': ['Low', 'Medium', 'High'],
        'Health': ['Poor', 'Good']
    }
)

model.add_cpds(
    attendance_cpd,
    sociallife_cpd,
    health_cpd,
    studytime_cpd,
    performance_cpd
)

print("Model Validation:", model.check_model())

plt.figure(figsize=(10, 6))
pos = {
    'Attendance': (0, 2),
    'SocialLife': (0, 0),
    'Health': (2, 0),
    'StudyTime': (2, 2),
    'Performance': (4, 1)
}

nx.draw(
    model,
    pos,
    with_labels=True,
    node_size=3500,
    node_color='lightblue',
    font_size=10,
    font_weight='bold',
    arrows=True,
    arrowsize=20
)

plt.title("Bayesian Network Graph")
plt.show()

def show_cpd(cpd):
    variables = cpd.variables
    target = variables[0]
    evidence = variables[1:]

    target_states = cpd.state_names[target]
    values = cpd.get_values()

    if evidence:
        evidence_states = [
            cpd.state_names[var] for var in evidence
        ]

        combinations = list(product(*evidence_states))

        columns = [
            " | ".join(combination)
            for combination in combinations
        ]
    else:
        columns = ["Probability"]

    table = pd.DataFrame(
        values,
        index=target_states,
        columns=columns
    )

    print("\nCPD Table:", target)
    print(table.round(3))

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 1.2), 3))
    ax.axis('off')

    ax.table(
        cellText=table.round(3).values,
        rowLabels=table.index,
        colLabels=table.columns,
        loc='center',
        cellLoc='center'
    )

    ax.set_title("CPD Table: " + target)
    plt.show()

for cpd in model.get_cpds():
    show_cpd(cpd)

inference = VariableElimination(model)

result = inference.query(
    variables=['Performance'],
    evidence={
        'StudyTime': 'High',
        'Health': 'Good'
    }
)

print("\nInference Result:")
print(result)

probabilities = result.values
performance_states = ['Poor', 'Average', 'Excellent']

for i in range(len(performance_states)):
    print(
        performance_states[i],
        ":",
        round(probabilities[i] * 100, 2),
        "%"
    )

best_performance = performance_states[probabilities.argmax()]

print("\nMost Likely Performance:", best_performance)
