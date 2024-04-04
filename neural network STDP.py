import numpy as np
import matplotlib.pyplot as plt
import nest


def raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,senders_teaching,ts_teaching,weights):
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
    plt.title('Spike Raster Plot - Lateral Ih Layer')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.grid()
    for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
        plt.vlines(spike_time, sender, sender + 1, color='violet')

    # plt.subplot(3, 2, 6)
    # plt.title('Spike Raster Plot - Teaching')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron ID')
    # plt.grid()
    # for sender, spike_time in zip(senders_lateral_ih_layer, ts_lateral_ih_layer):
    #     plt.vlines(spike_time, sender, sender + 1, color='pink')

    plt.subplot(3, 2, 6)
    plt.plot(weights, label='Layer 2 -> Layer 3', color='red')
    plt.xlabel('Simulation Step [1 step = 50 ms]')
    plt.ylabel('Synaptic Weight')
    plt.title('STDP Synaptic Weight Evolution')
    plt.grid()

    plt.tight_layout()
    plt.show()


def simulate_neural_network(num_steps=20, simulation_duration=50.0, min_current=378.0, max_current=380.0):
    # Reset the NEST simulator
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})

    # Create the neurons
    nest.SetDefaults("iaf_psc_alpha", {"I_e": 0.0})
    neuron_layer1 = nest.Create("iaf_psc_alpha", 2)
    neuron_layer2 = nest.Create("iaf_psc_alpha", 50)
    neuron_layer3 = nest.Create("iaf_psc_alpha", 50)
    noise_layer = nest.Create("sinusoidal_poisson_generator", 6)
    lateral_ih_layer = nest.Create("iaf_psc_alpha", 6)

    nest.SetStatus(noise_layer, {"rate": 10.0})  # Set the firing rate of the noisy neurons

    # Create spike recorders for each layer
    spike_recorder_layer1 = nest.Create("spike_recorder")
    spike_recorder_layer2 = nest.Create("spike_recorder")
    spike_recorder_layer3 = nest.Create("spike_recorder")
    spike_recorder_noise_layer = nest.Create("spike_recorder")
    spike_recorder_lateral_ih_layer = nest.Create("spike_recorder")
    teaching_layer = nest.Create('iaf_psc_alpha', 10)
    spike_recorder_teach =nest.Create('spike_recorder')
    # Connect the spike recorders to the neurons
    nest.Connect(neuron_layer1, spike_recorder_layer1)
    nest.Connect(neuron_layer2, spike_recorder_layer2)
    nest.Connect(neuron_layer3, spike_recorder_layer3)
    nest.Connect(noise_layer, spike_recorder_noise_layer)
    nest.Connect(lateral_ih_layer, spike_recorder_lateral_ih_layer)
    nest.Connect(teaching_layer,spike_recorder_teach)


    # Define connectivity between neurons
    syn_spec_l1l2 = {"weight": 1200.0}
    syn_spec_l1l3 = {"weight": 1200.0}
    syn_spec_l2l3 = {"weight": 1200.0}
    syn_spec_lnl2 = {"weight": 1200.0}
    syn_spec_l2ih = {"weight": 1200.0}
    syn_spec_ihl2 = {"weight": -13.0}
    syn_spec_l1lih = {"weight": -1200.0}

    # Connect Layer 1 to Layer 2
    # nest.Connect(neuron_layer1, neuron_layer2, syn_spec=syn_spec_l1l2)
    #
    # # Connect Layer 1 to Layer 3
    # nest.Connect(neuron_layer1, neuron_layer3, syn_spec=syn_spec_l1l3)
    #
    # # Connect Layer 2 to Layer 3
    # for i in range(50):
    #     nest.Connect(neuron_layer2[i], neuron_layer3[i], syn_spec=syn_spec_l2l3)
    #
    # # Connect the noisy neurons to specific neurons in layer 2
    # for i in range(10):
    #     nest.Connect(noise_layer[i], neuron_layer2[i * 5], syn_spec=syn_spec_lnl2)
    #
    # # Connect the lateral inhibition neurons to the same neurons as the noisy neurons
    # for i in range(10):
    #     nest.Connect(lateral_ih_layer[i], neuron_layer2[i * 5], syn_spec=syn_spec_l2ih)
    #
    # # Connect the noisy neurons to the lateral inhibition neurons
    # # for i in range(10):
    # #     nest.Connect(noise_layer[i], lateral_ih_layer[i], syn_spec=syn_spec_l1lih)



    for i in range(10):

        nest.Connect(teaching_layer[i],neuron_layer3[i],'one_to_one',syn_spec={'weight':1200})
    # for i in range(5):
    #     nest.Connect(neuron_layer1[0], teaching_layer[i], 'one_to_one', syn_spec={'weight': 1200, 'synapse_model': 'static_synapse'})

    for i in range(5):
        nest.Connect(neuron_layer1[1], teaching_layer[i], 'one_to_one', syn_spec={'weight': 1200, 'synapse_model': 'static_synapse'})


    for i in range(25):
        nest.Connect(neuron_layer1[0],neuron_layer2[i],'one_to_one',syn_spec={'weight':1200})



    for i in range(25,50):
        nest.Connect(neuron_layer1[1],neuron_layer2[i],'one_to_one',syn_spec={'weight':1200})

    #optimum weight is 64.98
    nest.Connect(neuron_layer2,neuron_layer3,'one_to_one',syn_spec={'weight':1200,'synapse_model':'static_synapse'})


    for i in range(6):
        nest.Connect(noise_layer[i],neuron_layer2[i])

  #  for i in range(6):
   #     nest.Connect(lateral_ih_layer[i],neuron_layer2[i],'one_to_one',syn_spec={'weight':-600})


    for i in range(5):
        nest.Connect(neuron_layer1[0],lateral_ih_layer[i],'one_to_one',syn_spec={'weight':1200})

    for i in range(5):
        nest.Connect(neuron_layer1[1],lateral_ih_layer[i],'one_to_one',syn_spec={'weight':1200})

    stdp_synapse_weights_l2l3=[]
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Generate random currents for neurons 1 and 2 in layer 1
        random_currents = np.random.uniform(min_current, max_current, size=2)


        # Apply the random currents to neurons in layer 1
        for i, current in enumerate(random_currents):
            nest.SetStatus(neuron_layer1[i], {"I_e": current})

        # Simulate the network for 50 ms
        nest.Simulate(simulation_duration)
        stdp_synapse_weights_l2l3.append(nest.GetStatus(nest.GetConnections(neuron_layer2, neuron_layer3), "weight"))

    # Retrieve spike times from spike recorders
    events_layer1 = nest.GetStatus(spike_recorder_layer1, "events")[0]
    events_layer2 = nest.GetStatus(spike_recorder_layer2, "events")[0]
    events_layer3 = nest.GetStatus(spike_recorder_layer3, "events")[0]
    events_noise_layer = nest.GetStatus(spike_recorder_noise_layer, "events")[0]
    events_lateral_ih_layer = nest.GetStatus(spike_recorder_lateral_ih_layer, "events")[0]
    events_teach = nest.GetStatus(spike_recorder_teach, "events")[0]

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

    senders_teching = events_lateral_ih_layer["senders"]
    ts_teaching = events_lateral_ih_layer["times"]


    # Call the function with the senders and ts
    raster_plot(senders_layer1, ts_layer1, senders_layer2, ts_layer2, senders_layer3, ts_layer3,
                senders_noise_layer, ts_noise_layer, senders_lateral_ih_layer, ts_lateral_ih_layer,  senders_teching,ts_teaching,stdp_synapse_weights_l2l3)


simulate_neural_network(num_steps=200, simulation_duration=50.0, min_current=300.0, max_current=450.0)