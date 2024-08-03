import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from PIL import Image
# Streamlit app
st.title("Full Adder Implementation with Machine Learning ")
inno=Image.open("Inno.png")
st.image(inno, use_column_width=False)
b=Image.open("Fulladder.png")
st.image(b)

# Load the trained models
model_sum_path = "sum.pkl"
model_cout_path = "Cout.pkl"

with open(model_sum_path, 'rb') as model_file:
    model_sum = pickle.load(model_file)

with open(model_cout_path, 'rb') as model_file:
    model_cout = pickle.load(model_file)

# Function to plot the full adder logic circuit dynamically
def plot_full_adder_circuit(a, b, carry_in):
    G = nx.DiGraph()

    # Calculate intermediate values
    xor1 = a ^ b
    xor2 = xor1 ^ carry_in
    and1 = a & b
    and2 = carry_in & xor1
    or_gate = and1 | and2

    # Node colors based on values
    color_map = {
        'A': 'green' if a == 1 else 'red',
        'B': 'green' if b == 1 else 'red',
        'Cin': 'green' if carry_in == 1 else 'red',
        'XOR1': 'green' if xor1 == 1 else 'red',
        'XOR2': 'green' if xor2 == 1 else 'red',
        'AND1': 'green' if and1 == 1 else 'red',
        'AND2': 'green' if and2 == 1 else 'red',
        'OR': 'green' if or_gate == 1 else 'red',
        'Sum': 'blue' if xor2 == 1 else 'grey',
        'Cout': 'blue' if or_gate == 1 else 'grey'
    }

    # Add nodes for inputs
    G.add_node('A', label='A', pos=(0, 1), color=color_map['A'])
    G.add_node('B', label='B', pos=(0, 0), color=color_map['B'])
    G.add_node('Cin', label='Cin', pos=(0, -1), color=color_map['Cin'])

    # Add nodes for gates
    G.add_node('XOR1', label=f'XOR1\n({xor1})', pos=(1, 0.5), color=color_map['XOR1'])
    G.add_node('XOR2', label=f'XOR2\n({xor2})', pos=(2, 0.5), color=color_map['XOR2'])
    G.add_node('AND1', label=f'AND1\n({and1})', pos=(1, -0.5), color=color_map['AND1'])
    G.add_node('AND2', label=f'AND2\n({and2})', pos=(2, -0.5), color=color_map['AND2'])
    G.add_node('OR', label=f'OR\n({or_gate})', pos=(3, 0), color=color_map['OR'])

    # Add final output nodes
    G.add_node('Sum', label=f'Sum\n({xor2})', pos=(4, 0.5), color=color_map['Sum'])
    G.add_node('Cout', label=f'Cout\n({or_gate})', pos=(4, -0.5), color=color_map['Cout'])

    # Add edges for connections
    G.add_edge('A', 'XOR1')
    G.add_edge('B', 'XOR1')
    G.add_edge('XOR1', 'XOR2')
    G.add_edge('Cin', 'XOR2')

    G.add_edge('A', 'AND1')
    G.add_edge('B', 'AND1')
    G.add_edge('XOR1', 'AND2')
    G.add_edge('Cin', 'AND2')

    G.add_edge('AND1', 'OR')
    G.add_edge('AND2', 'OR')

    # Add edges for final outputs
    G.add_edge('XOR2', 'Sum')
    G.add_edge('OR', 'Cout')

    # Define positions for nodes
    pos = nx.get_node_attributes(G, 'pos')
    colors = [nx.get_node_attributes(G, 'color').get(node, 'skyblue') for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=colors, font_size=10, font_weight='bold', arrows=True, arrowsize=20)

    # Draw node labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=12, font_weight='bold')

    plt.title("Full Adder Logic Circuit with Final Outputs")

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf

# Function to plot the DataFrame table
def plot_full_adder_table(df):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    # Save the table to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf

# Generate the full adder DataFrame
data = []
for a in [0, 1]:
    for b in [0, 1]:
        for carry_in in [0, 1]:
            sum_, carry_out = model_sum.predict([[a, b, carry_in]])[0], model_cout.predict([[a, b, carry_in]])[0]
            data.append([a, b, carry_in, sum_, carry_out])

columns = ['A', 'B', 'Cin', 'Sum', 'Cout']
df = pd.DataFrame(data, columns=columns)


# Input fields
a = st.selectbox("Input A", [0, 1])
b = st.selectbox("Input B", [0, 1])
carry_in = st.selectbox("Carry In", [0, 1])

# Button for prediction
if st.button("Submit"):
    # Prepare input for the model
    input_features = [[a, b, carry_in]]

    # Make predictions
    sum_pred = model_sum.predict(input_features)[0]
    cout_pred = model_cout.predict(input_features)[0]

    # Display the result
    st.write(f"**Predicted Sum (S):** {sum_pred}")
    st.write(f"**Predicted Carry-out (Cout):** {cout_pred}")

    # Plot and display the full adder logic circuit based on inputs
    st.write("### Full Adder Logic Circuit with Final Outputs")
    circuit_buf = plot_full_adder_circuit(a, b, carry_in)
    st.image(circuit_buf)

    # Plot and display the full adder DataFrame table
    st.write("### Full Adder Truth Table")
    table_buf = plot_full_adder_table(df)
    st.image(table_buf)
