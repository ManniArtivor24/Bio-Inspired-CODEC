import numpy as np
import matplotlib.pyplot as plt
import nest


def purpura_distance(spiketrain1, spiketrain2, cost):
    """
    spiketrain1 and spiketrain2 are vectors containing the spike times of two spike trains.
    cost is a parameter that determines the sensitivity of the distance calculation to the time differences between
    spikes.
    The function calculates the Victor-Purpura distance by iterating over the spike times of the two spike trains and
    summing exponential terms based on the time differences.
    The result is the Victor-Purpura distance, a measure of the dissimilarity between the spike trains.

    Parameters:
        spiketrain1 (array-like): Spike times of the first spike train.
        spiketrain2 (array-like): Spike times of the second spike train.
        cost (float): Cost parameter for the distance calculation.

    Returns:
        float: Victor-Purpura distance between the two spike trains.
    """
    # Ensure spiketrains are sorted
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)

    # Initialize the Victor-Purpura distance
    distance = 0.0

    # Calculate the Victor-Purpura distance
    i = 0
    j = 0
    while i < len(spiketrain1) and j < len(spiketrain2):
        time_diff = spiketrain1[i] - spiketrain2[j]
        if time_diff > 0:
            distance += np.exp(-time_diff / cost)
            j += 1
        elif time_diff < 0:
            distance += np.exp(time_diff / cost)
            i += 1
        else:
            i += 1
            j += 1

    # Add remaining spikes from longer spike train
    while i < len(spiketrain1):
        distance += np.exp(-(spiketrain1[i] - spiketrain2[-1]) / cost)
        i += 1
    while j < len(spiketrain2):
        distance += np.exp(-(spiketrain1[-1] - spiketrain2[j]) / cost)
        j += 1

    return distance


def rossum_metric(spiketrain1, spiketrain2, tau):
    """
    spiketrain1 and spiketrain2 are vectors containing the spike times of two spike trains.
    tau is a parameter that represents the time constant for the kernel function, which determines the weight of each
    spike time difference.
    The function calculates the van Rossum metric by iterating over all pairs of spike times from the two spike trains
    and summing the exponential decay of their absolute time differences.
    The result is the van Rossum metric, a measure of the similarity between the spike trains.

    Parameters:
        spiketrain1 (array-like): Spike times of the first spike train.
        spiketrain2 (array-like): Spike times of the second spike train.
        tau (float): Time constant for the kernel function.

    Returns:
        float: van Rossum metric between the two spike trains.
    """
    # Ensure spiketrains are sorted
    spiketrain1 = np.sort(spiketrain1)
    spiketrain2 = np.sort(spiketrain2)

    # Initialize the van Rossum distance
    distance = 0.0

    # Calculate the van Rossum metric
    for spike_time1 in spiketrain1:
        for spike_time2 in spiketrain2:
            distance += np.exp(-np.abs(spike_time1 - spike_time2) / tau)

    return distance


def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_teach_layer, ts_teach_layer, weights):
    plt.figure(figsize=(10, 8))

    # Layer 1
    plt.subplot(3, 2, 1)
    plt.title('Spike Raster Plot - Layer 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer1, ts_layer1):
        plt.vlines(spike_time, sender, sender + 1, color='red')

    # Layer 2
    plt.subplot(3, 2, 2)
    plt.title('Spike Raster Plot - Layer 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer2, ts_layer2):
        plt.vlines(spike_time, sender, sender + 1, color='blue')

    # Layer 3
    plt.subplot(3, 2, 3)
    plt.title('Spike Raster Plot - Layer 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_layer3, ts_layer3):
        plt.vlines(spike_time, sender, sender + 1, color='green')

    # Noise Layer
    plt.subplot(3, 2, 4)
    plt.title('Spike Raster Plot - Noise Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_noise_layer, ts_noise_layer):
        plt.vlines(spike_time, sender, sender + 1, color='orange')

    # Lateral Ih Layer
    plt.subplot(3, 2, 5)
    plt.title('Spike Raster Plot - Teach Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_teach_layer, ts_teach_layer):
        plt.vlines(spike_time, sender, sender + 1, color='pink')

    # STDP weights
    plt.subplot(3, 2, 6)
    plt.plot(weights, label='Layer 2 -> Layer 3', color='red')
    plt.xlabel('Simulation Step [1 step = 50 ms]')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    plt.savefig("results/Ex5_raster_plot.png")
    plt.show()


def simulate_neural_network(num_steps=20, simulation_duration=50.0, min_current=300.0, max_current=600.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})

    # Create the neurons
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})
    neuron_layer1 = nest.Create("iaf_psc_alpha", 100)
    neuron_layer2 = nest.Create("iaf_psc_alpha", 200)
    neuron_layer3 = nest.Create("iaf_psc_alpha", 200)
    noise_layer = nest.Create("poisson_generator", 2)
    lateral_ih_layer = nest.Create("iaf_psc_alpha", 2)
    teaching_layer = nest.Create("iaf_psc_alpha", 4)

    nest.SetStatus(noise_layer, {"rate": 0})  # Set the firing rate of the noisy neurons

    # Create spike recorders for each layer
    spike_recorder_layer1 = nest.Create("spike_recorder")
    spike_recorder_layer2 = nest.Create("spike_recorder")
    spike_recorder_layer3 = nest.Create("spike_recorder")
    spike_recorder_noise_layer = nest.Create("spike_recorder")
    spike_recorder_lateral_ih_layer = nest.Create("spike_recorder")
    spike_recorder_teaching = nest.Create("spike_recorder")

    # Connect the spike recorders to the neurons
    nest.Connect(neuron_layer1, spike_recorder_layer1)
    nest.Connect(neuron_layer2, spike_recorder_layer2)
    nest.Connect(neuron_layer3, spike_recorder_layer3)
    nest.Connect(noise_layer, spike_recorder_noise_layer)
    nest.Connect(lateral_ih_layer, spike_recorder_lateral_ih_layer)
    nest.Connect(teaching_layer, spike_recorder_teaching)

    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l2l3 = {"synapse_model": "static_synapse", "weight": 450.0}
    syn_spec_lnl2 = {"weight": 1200.0}
    syn_spec_tll3 = {"weight": 1200.0}

    # Define the connections for neuron 1 of layer 1 to neurons 1, 2, 3 of layer 2
    connections_layer1 = [(neuron_layer1[0], neuron_layer2[i]) for i in range(3)]

    # Define the connections for neuron 2 of layer 1 to neurons 4, 5, 6 of layer 2
    connections_layer2 = [(neuron_layer1[1], neuron_layer2[i]) for i in range(3, 6)]

    # Connect neuron 1 of layer 1 to neurons 1, 2, 3 of layer 2
    for connection in connections_layer1:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)

    # Connect neuron 2 of layer 1 to neurons 4, 5, 6 of layer 2
    for connection in connections_layer2:
        nest.Connect(connection[0], connection[1], syn_spec=syn_spec_l1l2)

    # connect all the neurons in layer 2 to layer 3
    nest.Connect(neuron_layer2, neuron_layer3, syn_spec=syn_spec_l2l3)

    # Connect teaching layer to layer 3 individually
    # nest.Connect(teaching_layer[0], neuron_layer3[0], syn_spec=syn_spec_tll3)
    # nest.Connect(teaching_layer[1], neuron_layer3[1], syn_spec=syn_spec_tll3)
    current_teach= [np.random.uniform(376,378 ,size=4)]
    for i in range(len(current_teach)):
        nest.SetStatus(teaching_layer[i],{'I_e':current_teach[i]})
    #
    # nest.Connect(teaching_layer[0],neuron_layer3[1],syn_spec={'weight':1200})
    # nest.Connect(teaching_layer[1], neuron_layer3[4],syn_spec={'weight':1200})
    # nest.Connect(teaching_layer[2], neuron_layer3[2],syn_spec={'weight':1200})
    # nest.Connect(teaching_layer[3], neuron_layer3[5],syn_spec={'weight':1200})

    # Define the connections from noisy neurons to layer 2 neurons
    connection_mapping = {0: [1, 4], 1: [2, 5]}

    # Connect the noisy neurons to specific neurons in layer 2
    for noise_neuron, target_neurons in connection_mapping.items():
        for target_neuron in target_neurons:
            nest.Connect(noise_layer[noise_neuron], neuron_layer2[target_neuron], syn_spec=syn_spec_lnl2)

    # Define synaptic weight recording for STDP connections
    stdp_synapse_weights_l2l3 = []

    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons 1 and 2 in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=2)

        # Apply the random currents to neurons in layer 1
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layer1[i], {"I_e": current})
            # nest.SetStatus(teaching_layer[i], {"I_e": current})

        # Simulate the network for 50 ms
        nest.Simulate(simulation_duration)
        stdp_synapse_weights_l2l3.append(nest.GetStatus(nest.GetConnections(neuron_layer2, neuron_layer3), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorder_layer1, "events")[0]
    events_layer2 = nest.GetStatus(spike_recorder_layer2, "events")[0]
    events_layer3 = nest.GetStatus(spike_recorder_layer3, "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorder_noise_layer, "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorder_lateral_ih_layer, "events")[0]
    events_teaching_layer = nest.GetStatus(spike_recorder_teaching, "events")[0]

    # Extract senders and spike times
    senders_layer1 = events_layer1["senders"]
    ts_layer1 = events_layer1["times"]

    senders_layer2 = events_layer2["senders"]
    ts_layer2 = events_layer2["times"]

    senders_layer3 = events_layer3["senders"]
    ts_layer3 = events_layer3["times"]

    senders_noise_layer = events_noise_layer["senders"]
    ts_noise_layer = events_noise_layer["times"]

    senders_lateral_ih_layer = events_lateral_ih_layer["senders"]
    ts_lateral_ih_layer = events_lateral_ih_layer["times"]

    senders_teaching_layer = events_teaching_layer["senders"]
    ts_teaching_layer = events_teaching_layer["times"]

    # Compute the purpura_distance and rossum_metric
    cost = 1.0  # You can adjust this parameter as needed
    tau = 1.0  # You can adjust this parameter as needed

    # Extract spike times for neurons n0 and n1 in Layer 1 and Layer 3
    n0_spikes_layer1 = events_layer1["times"][events_layer1["senders"] == neuron_layer1[0]]
    n1_spikes_layer1 = events_layer1["times"][events_layer1["senders"] == neuron_layer1[1]]
    n0_spikes_layer3 = events_layer3["times"][events_layer3["senders"] == neuron_layer3[0]]
    n1_spikes_layer3 = events_layer3["times"][events_layer3["senders"] == neuron_layer3[1]]

    purpura_distance_n0 = purpura_distance(n0_spikes_layer1, n0_spikes_layer3, cost)
    purpura_distance_n1 = purpura_distance(n1_spikes_layer1, n1_spikes_layer3, cost)

    rossum_metric_n0 = rossum_metric(n0_spikes_layer1, n0_spikes_layer3, tau)
    rossum_metric_n1 = rossum_metric(n1_spikes_layer1, n1_spikes_layer3, tau)

    print("Purpura Distance between n0 in Layer 1 and n0 in Layer 3:", purpura_distance_n0)
    print("Purpura Distance between n1 in Layer 1 and n1 in Layer 3:", purpura_distance_n1)
    print("Rossum Metric between n0 in Layer 1 and n0 in Layer 3:", rossum_metric_n0)
    print("Rossum Metric between n1 in Layer 1 and n1 in Layer 3:", rossum_metric_n1)

    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_teaching_layer, ts_teaching_layer,
                stdp_synapse_weights_l2l3)


simulate_neural_network(num_steps=120, simulation_duration=50.0, min_current=400.0, max_current=405.0)