#!/bin/python
import itertools as it
from pyNN.utility import get_simulator, init_logging, normalized_filename
from quantities import ms
from random import randint
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from datetime import datetime as dt
import neo
import numpy as np
import sys
import os
import imageio
sys.path.append("/Users/kevin/Documents/SI5/Cours/T.E.R/Code/Amelie/EvData/translate_2_formats")
from events2spikes import ev2spikes

start = dt.now()

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--nb-convolution", "The number of convolution layers of the model", {"action": "store", "type": int}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

if sim == "nest":
    from pyNN.nest import *

sim.setup(timestep=0.01)

# Constants, variables
NB_CONV_LAYERS = options.nb_convolution
if NB_CONV_LAYERS < 2:
    print("The number of convolution layers should be at least 2")
    quit()

DIRECTIONS = {-1: "INDETERMINATE", 0: "SOUTH-WEST ↙︎", 1: "SOUTH-EAST ↘︎", 2: "NORTH-EAST ↗︎", 3: "NORTH-WEST ↖︎"} # KEY=DIRECTIONS ID ; VALUE=STRING REPRESENTING THE DIRECTION
NB_DIRECTIONS = min(len(DIRECTIONS)-1, NB_CONV_LAYERS) # No more than available directions, and at least 2 directions. -1 to ignore INDETERMINATE
OUTPUT_PATH_GENERIC = "./output"
SIZE_CONV_FILTER = 5

### Generate input data

time_data = 30_000 # TODO de base 1e6, à réduire pour tests
temporal_reduction = 1_000
pattern_interval = 100
pattern_duration = 5
num = time_data//pattern_interval

# The input should be at least 13*13 for a duration of 5 since we want to leave a margin of 4 neurons on the edges when generating data
x_input = 13
filter_x = 5
x_output = x_input - filter_x + 1

y_input = 13
filter_y = 5
y_output = y_input - filter_y + 1

x_margin = y_margin = 4

# Dataset Generation

input_events = np.zeros((0,4)) # 4 because : x, y, polarity, time
for t in range(int(time_data/pattern_interval)):
    direction = randint(0, NB_DIRECTIONS-1) # {NB_DIRECTIONS} directions
    if direction == 0:
        start_x = randint(x_margin, x_input-pattern_duration-x_margin) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((input_events, [[start_x+d, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]), axis=0)
    
    elif direction == 1:
        start_x = randint(x_input-x_margin-1, x_input-pattern_duration) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((input_events, [[start_x-d, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]), axis=0)

    elif direction == 2:
        start_x = randint(x_input-x_margin-1, x_input-pattern_duration) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_input-y_margin-1, y_input-pattern_duration)
        input_events = np.concatenate((input_events, [[start_x-d, start_y-d, 1, d+t*pattern_interval] for d in range(pattern_duration)]), axis=0)

    elif direction == 3:
        start_x = randint(x_margin, x_input-pattern_duration-x_margin) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_input-y_margin-1, y_input-pattern_duration)
        input_events = np.concatenate((input_events, [[start_x+d, start_y-d, 1, d+t*pattern_interval] for d in range(pattern_duration)]), axis=0)

input_spiketrain, _, _ = ev2spikes(input_events, width=x_input, height=y_input)


### Build Network 

# Populations

Input = sim.Population(
    x_input*y_input,  
    sim.SpikeSourceArray(spike_times=input_spiketrain), 
    label="Input"
)
Input.record("spikes")

Convolutions_parameters = {
    'tau_m': 20.0,       # membrane time constant (in ms)   
    'tau_refrac': 30.0,  # duration of refractory period (in ms) 0.1 de base
    'v_reset': -70.0,    # reset potential after a spike (in mV) 
    'v_rest': -70.0,     # resting membrane potential (in mV)
    'v_thresh': -5.0,    # spike threshold (in mV) -5 de base
}

# The size of a convolution layer with a filter of size x*y is input_x-x+1 * input_y-y+1
ConvLayers = []
for i in range(NB_CONV_LAYERS):
    Conv_i = sim.Population(
        x_output*y_output, 
        sim.IF_cond_exp(**Convolutions_parameters),
    )
    Conv_i.record(('spikes','v'))
    ConvLayers.append(Conv_i)

# List connector

weight_N = 0.35 
delays_N = 15.0 
weight_teta = 0.005 
delays_teta = 0.05 

Weight_conv = np.random.normal(weight_N, weight_teta, size=(NB_CONV_LAYERS, filter_x, filter_y))
Delay_conv =  np.random.normal(delays_N, delays_teta, size=(NB_CONV_LAYERS, filter_x, filter_y))

Input_to_output_i_conn = [[] for _ in range(NB_CONV_LAYERS)]
c = 0

for in2out_conn in Input_to_output_i_conn:

    for X,Y in it.product(range(x_output), range(y_output)):

        idx = np.ravel_multi_index( (X,Y) , (x_output, y_output) )

        conn = []
        for x, y in it.product(range(filter_x), range(filter_y)):
            w = Weight_conv[c, x, y]
            d = Delay_conv[ c, x, y]
            A = np.ravel_multi_index( (X+x,Y+y) , (x_input, y_input) )
            conn.append( ( A, idx, w, d ) )

        in2out_conn += conn
    
    c += 1

# Projections

Input_to_Conv_i = []
for i in range(NB_CONV_LAYERS):
    Input_to_Conv_i.append(
        sim.Projection(
            Input, ConvLayers[i],
            connector = sim.FromListConnector(Input_to_output_i_conn[i]),
            synapse_type = sim.StaticSynapse(),
            receptor_type = 'excitatory',
            label = 'Input to Conv'+str((i+1))
        )
    )

# Establish Lateral Inhibition

for idx_a in range(NB_CONV_LAYERS):
    for idx_b in range(NB_CONV_LAYERS):
        if idx_a != idx_b: # Avoid creating Lateral inhibition with himself
            Conv_rising_to_conv_falling = sim.Projection(
                ConvLayers[idx_a], ConvLayers[idx_b],
                connector = sim.OneToOneConnector(),
                synapse_type = sim.StaticSynapse(
                    weight = 50,
                    delay = 0.01
                ),
                receptor_type = "inhibitory",
                label = "Lateral inhibition - Conv"+str(idx_a+1)+" to Conv"+str(idx_b+1)
            )


# We will use this list to know which convolution layer has reached its stop condition
full_stop_condition= [False] * NB_CONV_LAYERS
# Each filter of each convolution layer will be put in this list and actualized at each stimulus
final_filters = [[] for _ in range(NB_CONV_LAYERS)]
# Sometimes, even with lateral inhibition, two neurons on the same location in different convolution
# layers will both spike (due to the minimum delay on those connections). So we keep track of
# which neurons in each layer has already spiked for this stimulus. (Everything is put back to False at the end of the stimulus)
neuron_activity_tag = [[False for _ in range((x_input-filter_x+1)*(y_input-filter_y+1))] for _ in range(NB_CONV_LAYERS)]


### Run simulation

# Callback classes

class LastSpikeRecorder(object):

    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        self.global_spikes = [[] for _ in range(self.population.size)]
        self.annotations = {}
        self.final_spikes = []


        if type(self.population) != list:
            self._spikes = np.ones(self.population.size) * (-1)
        else:
            self._spikes = np.ones(len(self.population)) * (-1)

    def __call__(self, t):
        if t > 0:
            if type(self.population) != list:
                population_spikes = self.population.get_data("spikes", clear=True).segments[0].spiketrains
                self._spikes = map(
                    lambda x: x[-1].item() if len(x) > 0 else -1, 
                    population_spikes
                )
                self._spikes = np.fromiter(self._spikes, dtype=float)

                if t == self.interval:
                    for n, neuron_spikes in enumerate(population_spikes):
                        self.annotations[n] = neuron_spikes.annotations

            else:
                self._spikes = []
                for subr in self.population:
                    sp = subr.get_data("spikes", clear=True).segments[0].spiketrains
                    spikes_subr = map(
                        lambda x: x[-1].item() if len(x) > 0 else -1, 
                        sp
                    )
                    self._spikes.append(max(spikes_subr))

            assert len(self._spikes) == len(self.global_spikes)
            if len(np.unique(self._spikes)) > 1:
                idx = np.where(self._spikes != -1)[0]
                for n in idx:
                    self.global_spikes[n].append(self._spikes[n])

        return t+self.interval

    def get_spikes(self):
        for n, s in enumerate(self.global_spikes):
            self.final_spikes.append( neo.core.spiketrain.SpikeTrain(s*ms, t_stop=time_data, **self.annotations[n]) )
        return self.final_spikes

class WeightDelayRecorder(object):

    def __init__(self, sampling_interval, proj):
        self.interval = sampling_interval
        self.projection = proj

        self.weight = None
        self._weights = []
        self.delay = None
        self._delays = []

    def __call__(self, t):
        attribute_names = self.projection.synapse_type.get_native_names('weight','delay')
        self.weight, self.delay = self.projection._get_attributes_as_arrays(attribute_names, multiple_synapses='sum')
        
        self._weights.append(self.weight)
        self._delays.append(self.delay)

        return t+self.interval

    def update_weights(self, w):
        assert self._weights[-1].shape == w.shape
        self._weights[-1] = w

    def update_delays(self, d):
        assert self._delays[-1].shape == d.shape
        self._delays[-1] = d

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms, name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal

    def get_weights(self):
        signal = neo.AnalogSignal(self._delays, units='ms', sampling_period=self.interval * ms, name="delay")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._delays[0])))
        return signal


class visualiseTime(object):


    def __init__(self, sampling_interval):
        self.interval = sampling_interval
        self.times_called = 0
        self.delay_and_weight_evolution_plot = []
        self.OUTPUT_PATH_CURRENT_RUN = OUTPUT_PATH_GENERIC + '/' + dt.now().strftime("%Y%m%d-%H%M%S")
        if options.plot_figure: # TODO find a better way, actually 2 conditions based on this
            if not os.path.isdir(OUTPUT_PATH_GENERIC): # Output folder
                os.mkdir(OUTPUT_PATH_GENERIC)

            if not os.path.isdir(self.OUTPUT_PATH_CURRENT_RUN): # Folder for current run, inside Output folder
                os.mkdir(self.OUTPUT_PATH_CURRENT_RUN)


    def __call__(self, t):
        print("step : {}".format(t))
        if full_stop_condition[0] and full_stop_condition[1]:
            print("!!!! FINISHED LEARNING !!!!") 
            sim.end()
            self.print_filters(t) # TODO only print when reaching final condition
            # exit() # TODO necessary ? because with this we skip the end part of the code with plots and more...
        if t > 1 and int(t) % pattern_interval==0:
            self.print_filters(t)

        return t + self.interval


    def recognize_movement(self, delay_matrix):
        """
        Return an int that indicates the direction in which the input delay matrix has specialized.
        For the moment, 5 possibles outputs:
        - 4 diagonals : NORTH-EAST (2) ; SOUTH-EAST (1) ; SOUTH-WEST (3) ; NORTH-WEST (0)
        - 1 no specialization : INDETERMINATE (-1)
        """
        def pred_movement(delay_matrix, rangeI, rangeJ, prevI, prevJ):
            delay_matrix = delay_matrix.copy()
            ignore_idx = [(rangeI[0]+prevI, rangeJ[0]+prevJ)]
            for i, j in zip(rangeI, rangeJ):
                ignore_idx.append((i, j))
                if delay_matrix[i+prevI][j+prevJ] < delay_matrix[i][j]:
                    return False

            first_mvt_idx = ignore_idx[0]
            first_mvt_idx_val = delay_matrix[first_mvt_idx[0]][first_mvt_idx[1]]

            for idx in ignore_idx:
                delay_matrix[idx[0]][idx[1]] = np.inf

            if not np.amin(delay_matrix) > first_mvt_idx_val: # S'assurer que le délai min en dehors de la diagonale soit supérieur au délai du premier pixel du mouvement # TODO et ce, avec un écart d'au moins 0.2 ms
                return False
            return True

        size_matrix = len(delay_matrix)
        if pred_movement(delay_matrix, range(1, size_matrix), range(1, size_matrix), -1, -1):
            return 1 # HGBD
        elif pred_movement(delay_matrix, range(size_matrix-2, -1, -1), range(size_matrix-2, -1, -1), 1, 1):
            return 3 # BDHG
        elif pred_movement(delay_matrix, range(1, size_matrix), range(size_matrix-2, -1, -1), -1, 1):
            return 0 # HDBG
        elif pred_movement(delay_matrix, range(size_matrix-2, -1, -1), range(1, size_matrix), 1, -1):
            return 2 # BGHD
        else:
            return -1


    def print_filters(self, t):
        """
        Create and save a plot that contains for each convolution filter its delay matrix and associated weights of the current model state
        """
        SAVED_FILE_NAME = self.OUTPUT_PATH_CURRENT_RUN + '/delays_and_weights_'+str(self.times_called)+".png"
        LOG_STR = ["Delays of convolution", "Weights of convolution"]
        COLOR_MAP_TYPE = plt.cm.autumn # https://matplotlib.org/stable/tutorials/colors/colormaps.html

        SCALING_VALUE = SIZE_CONV_FILTER + ((NB_CONV_LAYERS - 2) * 4) # Make sure the plot is big enought depending on number of convolution used
        FONTSIZE = 9+(1.05*NB_CONV_LAYERS)

        fig, axs = plt.subplots(nrows=len(LOG_STR), ncols=NB_CONV_LAYERS, sharex=True, figsize=(SCALING_VALUE, SCALING_VALUE))
        for i in range(len(LOG_STR)): # Delay and Weight
            for layer_n in range(NB_CONV_LAYERS): # The number of convolution layer in the model
                data = final_filters[layer_n][i]
                title = LOG_STR[i] + ' ' + str(layer_n)
                if i == 0: # Delay matrix part
                    movement_id = self.recognize_movement(data)
                    title += '\n' + DIRECTIONS[movement_id]
                curr_matrix = axs[i][layer_n]
                curr_matrix.set_title(title, fontsize=FONTSIZE)
                im = curr_matrix.imshow(data, cmap=COLOR_MAP_TYPE)
                fig.colorbar(im, ax=curr_matrix, fraction=0.046, pad=0.04) # https://stackoverflow.com/a/26720422
        fig.suptitle('Delays and Weights kernel at t:'+str(t), fontsize=FONTSIZE)
        plt.tight_layout()
        fig.savefig(SAVED_FILE_NAME)
        plt.close() # Avoid getting displayed at the end

        self.delay_and_weight_evolution_plot.append(SAVED_FILE_NAME)
        print("[", str(self.times_called) , "] : Images of delays and weights saved as", SAVED_FILE_NAME)
        self.times_called += 1


    def print_final_filters(self):
        """
        Create a gif containing every images generated by print_filters
        """
        imgs = [imageio.imread(step_file) for step_file in self.delay_and_weight_evolution_plot]
        imageio.mimsave(self.OUTPUT_PATH_CURRENT_RUN + '/delays_and_weights_evolution.gif', imgs, duration=1) # 1s between each frame of the gif


class NeuronReset(object):
    """    
    Resets neuron_activity_tag to False for all neurons in all layers.
    Also injects a negative amplitude pulse to all neurons at the end of each stimulus
    So that all membrane potentials are back to their resting values.
    """

    def __init__(self, sampling_interval, pops):
        self.interval = sampling_interval
        self.populations = pops 

    def __call__(self, t):
        for conv in neuron_activity_tag:
            for cell in range(len(conv)):
                conv[cell] = False

        if t > 0:
            print("!!! RESET !!!")
            if type(self.populations)==list:
                for pop in self.populations:
                    pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+10)
                    pulse.inject_into(pop)
            else:
                pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+10)
                pulse.inject_into(self.populations)

            self.interval = pattern_interval
        return t + self.interval


class InputClear(object):
    """
    When called, simply gets the data from the input with the 'clear' parameter set to True.
    By periodically clearing the data from the populations the simulation goes a lot faster.
    """

    def __init__(self, sampling_interval, pops_to_clear_data):
        self.interval = sampling_interval
        self.pop_clear = pops_to_clear_data

    def __call__(self, t):
        if t > 0:
            print("!!! INPUT CLEAR !!!")
            try:
                input_spike_train = self.pop_clear.get_data("spikes", clear=True).segments[0].spiketrains 
            except:
                pass
            self.interval = pattern_interval
        return t + self.interval


class LearningMechanisms(object):
    def __init__(
        self, 
        sampling_interval, 
        input_spikes_recorder, output_spikes_recorder,
        projection, projection_delay_weight_recorder,
        B_plus, B_minus, 
        tau_plus, tau_minus, 
        A_plus, A_minus, 
        teta_plus, teta_minus, 
        filter_d, filter_w, 
        stop_condition, 
        growth_factor, 
        Rtarget=0.005, 
        lamdad=0.002, lamdaw=0.00005, 
        thresh_adapt=True, 
        label=0
    ):
        print("PROJECT:", projection)
        self.interval = sampling_interval
        self.projection = projection
        self.input = projection.pre
        self.output = projection.post

        self.input_spikes = input_spikes_recorder 
        self.output_spikes = output_spikes_recorder
        self.DelayWeights = projection_delay_weight_recorder
        
        # We keep the last time of spike of each neuron
        self.input_last_spiking_times = self.input_spikes._spikes
        self.output_last_spiking_times = self.output_spikes._spikes
        
        self.B_plus = B_plus
        self.B_minus = B_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.max_delay = False # If set to False, we will find the maximum delay on first call.
        self.filter_d = filter_d
        self.filter_w = filter_w
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.teta_plus = teta_plus
        self.teta_minus = teta_minus
        self.c = stop_condition
        self.growth_factor = growth_factor
        self.label = label
        self.thresh_adapt=thresh_adapt
        
        # For each neuron, we count their number of spikes to compute their activation rate.
        self.total_spike_count_per_neuron = [0 for _ in range(len(self.output))] 
        
        # Number of times this has been called.
        self.call_count = 0 
        
        self.Rtarget = Rtarget
        self.lamdaw = lamdaw 
        self.lamdad = lamdad

    def __call__(self, t):

        if t == 0 :
            print("No data")
            return t + pattern_interval

        self.call_count += 1
        final_filters[self.label] = [self.filter_d, self.filter_w]

        # The sum of all homeostasis delta_d and delta_t computed for each cell
        homeo_delays_total = 0
        homeo_weights_total = 0

        # Since we can't increase the delays past the maximum delay set at the beginning of the simulation,
        # we find the maximum delay during the first call
        if self.max_delay == False:
            self.max_delay = 0.01
            for x in self.DelayWeights.delay:
                for y in x:
                    if not np.isnan(y) and y > self.max_delay:
                        self.max_delay = y

        for pre_neuron in range(self.input.size):
            if self.input_spikes._spikes[pre_neuron] != -1 and self.input_spikes._spikes[pre_neuron] > self.input_last_spiking_times[pre_neuron]:
                # We actualize the last time of spike for this neuron
                self.input_last_spiking_times[pre_neuron] = self.input_spikes._spikes[pre_neuron]
                print("PRE SPIKE {} : {}".format(pre_neuron, self.input_spikes._spikes[pre_neuron]))

        #print("OOOOLLL", self.output_spikes._spikes)
        #print("OOOMMM", self.output_spikes.global_spikes)
        for post_neuron in range(self.output.size):
            if self.output_spikes._spikes[post_neuron] != -1 and self.check_activity_tags(post_neuron):
                neuron_activity_tag[self.label][post_neuron] = True
                print("***** STIMULUS {} *****".format(t//pattern_interval))

                self.total_spike_count_per_neuron[post_neuron] += 1

                # The neuron spiked during this stimulus and its threshold should be increased.
                # Since Nest won't allow neurons with a threshold > 0 to spike, we decrease v_rest instead.
                current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                if self.thresh_adapt:
                    self.output.__getitem__(post_neuron).v_rest=current_rest-(1.0-self.Rtarget)
                    self.output.__getitem__(post_neuron).v_reset=current_rest-(1.0-self.Rtarget)
                print("=== Neuron {} from layer {} spiked ! Whith rest = {} ===".format(post_neuron, self.label, current_rest))
                print("Total pikes of neuron {} from layer {} : {}".format(post_neuron, self.label, self.total_spike_count_per_neuron[post_neuron]))

                if self.output_spikes._spikes[post_neuron] > self.output_last_spiking_times[post_neuron] and not self.stop_condition(post_neuron):
                    # We actualize the last time of spike for this neuron
                    self.output_last_spiking_times[post_neuron] = self.output_spikes._spikes[post_neuron]

                    # We now compute a new delay for each of its connections using STDP
                    print("TAILLE PRE_NEURON", len(self.DelayWeights.delay))
                    for pre_neuron in range(len(self.DelayWeights.delay)):
                        
                        # For each post synaptic neuron that has a connection with pre_neuron, we also check that both neurons
                        # already spiked at least once.
                        if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and not np.isnan(self.DelayWeights.weight[pre_neuron][post_neuron]) and self.input_last_spiking_times[pre_neuron] != -1 and self.output_last_spiking_times[post_neuron] != -1:

                            # Some values here have a dimension in ms
                            delta_t = self.output_last_spiking_times[post_neuron] - self.input_last_spiking_times[pre_neuron] - self.DelayWeights.delay[pre_neuron][post_neuron]
                            delta_d = self.G(delta_t)
                            delta_w = self.F(delta_t)

                            print("STDP from layer: {} with post_neuron: {} and pre_neuron: {} deltad: {}, deltat: {}".format(self.label, post_neuron, pre_neuron, delta_d*ms, delta_t*ms))
                            print("TIME PRE {} : {} TIME POST 0: {} DELAY: {}".format(pre_neuron, self.input_last_spiking_times[pre_neuron], self.output_last_spiking_times[post_neuron], self.DelayWeights.delay[pre_neuron][post_neuron]))
                            self.actualize_filter(pre_neuron, post_neuron, delta_d, delta_w)
            else:
                # The neuron did not spike and its threshold should be lowered

                if self.thresh_adapt:
                    current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                    self.output.__getitem__(post_neuron).v_rest=current_rest+self.Rtarget
                    self.output.__getitem__(post_neuron).v_reset=current_rest+self.Rtarget

            # Homeostasis regulation per neuron
            Robserved = self.total_spike_count_per_neuron[post_neuron]/self.call_count
            K = (self.Rtarget - Robserved)/self.Rtarget
            delta_d = -self.lamdad*K
            delta_w = self.lamdaw*K
            homeo_delays_total += delta_d  
            homeo_weights_total += delta_w 
            print("Rate of neuron {} from layer {}: {}".format(post_neuron, self.label, Robserved))


        print("****** CONVO {} homeo_delays_total: {}, homeo_weights_total: {}".format(self.label, homeo_delays_total, homeo_weights_total))
        self.actualizeAllFilter( homeo_delays_total+self.growth_factor*self.interval, homeo_weights_total)

        # At last we give the new delays and weights to our projections
        
        self.DelayWeights.update_delays(self.DelayWeights.delay)
        self.DelayWeights.update_weights(self.DelayWeights.weight)
        
        ### HERE MODIFY 
        #print("\n\nIDK:", self.projection, " ET ", self.DelayWeights.delay)
        #print("IDK2", self.DelayWeights.delay.shape, type(self.DelayWeights.delay))
        #self.projection.set(delay = np.ones((169, 81)) * 16) 
        self.projection.set(delay = self.DelayWeights.delay)
        self.projection.set(weight = self.DelayWeights.weight)

        # We update the list that tells if this layer has finished learning the delays and weights
        full_stop_condition[self.label] = self.full_stop_check()
        return t + pattern_interval

    # Computes the delay delta by applying the STDP
    def G(self, delta_t):
        if delta_t >= 0:
            delta_d = -self.B_minus*np.exp(-delta_t/self.teta_minus)
        else:
            delta_d = self.B_plus*np.exp(delta_t/self.teta_plus)
        return delta_d

    # Computes the weight delta by applying the STDP
    def F(self, delta_t):
        if delta_t >= 0:
            delta_w = self.A_plus*np.exp(-delta_t/self.tau_plus)
        else:
            delta_w = -self.A_minus*np.exp(delta_t/self.tau_minus)
        return delta_w

    # Given a post synaptic cell, returns if that cell has reached its stop condition for learning
    def stop_condition(self, post_neuron):
        for pre_neuron in range(len(self.DelayWeights.delay)):
            if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] <= self.c:
                return True
        return False

    # Checks if all cells have reached their stop condition
    def full_stop_check(self):
        for post_neuron in range(self.output.size):
            if not self.stop_condition(post_neuron):
                return False
        return True

    # Applies the current weights and delays of the filter to all the cells sharing those
    def actualize_filter(self, pre_neuron, post_neuron, delta_d, delta_w):
        # We now find the delay/weight to use by looking at the filter
        convo_coords = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
        input_coords = [pre_neuron%x_input, pre_neuron//x_input]
        filter_coords = [input_coords[0]-convo_coords[0], input_coords[1]-convo_coords[1]]

        # And we actualize delay/weight of the filter after the STDP
        #print(pre_neuron, post_neuron)
        #print(np.unravel_index(pre_neuron, (13,13)))
        #print(np.unravel_index(post_neuron, (9,9)))
        #print(input_coords, convo_coords, filter_coords)
        #print(self.filter_d.shape, self.filter_w.shape)
        self.filter_d[filter_coords[0]][filter_coords[1]] = max(0.01, min(self.filter_d[filter_coords[0]][filter_coords[1]]+delta_d, self.max_delay))
        self.filter_w[filter_coords[0]][filter_coords[1]] = max(0.05, self.filter_w[filter_coords[0]][filter_coords[1]]+delta_w)

        # Finally we actualize the weights and delays of all neurons that use the same filter
        for window_x in range(0, x_input - (filter_x-1)):
            for window_y in range(0, y_input - (filter_y-1)):
                input_neuron_id = window_x+filter_coords[0] + (window_y+filter_coords[1])*x_input
                convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
                if not np.isnan(self.DelayWeights.delay[input_neuron_id][convo_neuron_id]) and not np.isnan(self.DelayWeights.weight[input_neuron_id][convo_neuron_id]):
                    self.DelayWeights.delay[input_neuron_id][convo_neuron_id] = self.filter_d[filter_coords[0]][filter_coords[1]]
                    self.DelayWeights.weight[input_neuron_id][convo_neuron_id] = self.filter_w[filter_coords[0]][filter_coords[1]]

    # Applies delta_d and delta_w to the whole filter 
    def actualizeAllFilter(self, delta_d, delta_w):

        for x in range(len(self.filter_d)):
            for y in range(len(self.filter_d[x])):
                self.filter_d[x][y] = max(0.01, min(self.filter_d[x][y]+delta_d, self.max_delay))
                self.filter_w[x][y] = max(0.05, self.filter_w[x][y]+delta_w)

        # Finally we actualize the weights and delays of all neurons that use the same filter
        for window_x in range(0, x_input - (filter_x-1)):
            for window_y in range(0, y_input - (filter_y-1)):
                for x in range(len(self.filter_d)):
                    for y in range(len(self.filter_d[x])):
                        input_neuron_id = window_x+x + (window_y+y)*x_input
                        convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
                        if input_neuron_id < self.input.size and not np.isnan(self.DelayWeights.delay[input_neuron_id][convo_neuron_id]) and not np.isnan(self.DelayWeights.weight[input_neuron_id][convo_neuron_id]):
                            self.DelayWeights.delay[input_neuron_id][convo_neuron_id] = self.filter_d[x][y]
                            self.DelayWeights.weight[input_neuron_id][convo_neuron_id] = self.filter_w[x][y]

    def get_filters(self):
        return self.filter_d, self.filter_w

    def check_activity_tags(self, neuron_to_check):
        for conv in neuron_activity_tag:
            if conv[neuron_to_check]:
                return False
        return True


### Simulation parameters

growth_factor = (0.001/pattern_interval)*pattern_duration # <- juste faire *duration dans STDP We increase each delay by this constant each step

# Stop Condition
c = 1.0

# STDP weight
A_plus = 0.05  
A_minus = 0.05
tau_plus= 1.0 
tau_minus= 1.0

# STDP delay (2.5 is good too)
B_plus = 5.0 
B_minus = 5.0
teta_plus = 1.0 
teta_minus = 1.0

STDP_sampling = pattern_interval

### Launch simulation

visu = visualiseTime(sampling_interval=500)
wd_rec = WeightDelayRecorder(sampling_interval=1, proj=Input_to_Conv_i[0])

Input_spikes = LastSpikeRecorder(sampling_interval=STDP_sampling, pop=Input)
Conv_i_spikes = []
Input_to_conv_i_delay_weight = []
for i in range(NB_CONV_LAYERS):
    Conv_i_spikes.append(LastSpikeRecorder(sampling_interval=STDP_sampling, pop=ConvLayers[i]))
    Input_to_conv_i_delay_weight.append(WeightDelayRecorder(sampling_interval=STDP_sampling, proj=Input_to_Conv_i[i]))

neuron_reset = NeuronReset(sampling_interval=pattern_interval-15, pops=ConvLayers) # TODO give a copy ? previous: pops=[Conv1, Conv2]
# input_clear = InputClear(sampling_interval=pattern_interval+1, pops_to_clear_data=Input)

Learn_i = []
for i in range(NB_CONV_LAYERS):
    Learn_i.append(LearningMechanisms(sampling_interval=STDP_sampling, 
    input_spikes_recorder=Input_spikes, 
    output_spikes_recorder=Conv_i_spikes[i], 
    projection=Input_to_Conv_i[i], 
    projection_delay_weight_recorder=Input_to_conv_i_delay_weight[i], 
    B_plus=B_plus, 
    B_minus=B_minus, 
    tau_plus=tau_plus, 
    tau_minus=tau_minus, 
    filter_d=Delay_conv[i], 
    A_plus=A_plus, 
    A_minus=A_minus, 
    teta_plus=teta_plus, 
    teta_minus=teta_minus, 
    filter_w=Weight_conv[i] , 
    stop_condition=c, 
    growth_factor=growth_factor, 
    label=i))

callback_list = [visu, wd_rec, Input_spikes, *Conv_i_spikes, *Input_to_conv_i_delay_weight, *Learn_i, neuron_reset]
sim.run(time_data, callbacks=callback_list)

print("complete simulation run time:", dt.now() - start)

### Plot figure

if options.plot_figure :

    extension = '_'+str(NB_DIRECTIONS)+'directions_'+str(NB_CONV_LAYERS)+'filters'
    title = "Delay learning - "+ str(NB_DIRECTIONS)+ " directions - "+str(NB_CONV_LAYERS)+" filters"
    
    Conv_i_data = [conv_i.get_data().segments[0] for conv_i in ConvLayers]

    Input_spikes = Input_spikes.get_spikes()
    Conv_i_spikes = [conv_i_spikes.get_spikes() for conv_i_spikes in Conv_i_spikes]

    figure_filename = normalized_filename("Results", "delay_learning"+extension, "png", options.simulator)

    figure_params = []
    # Add reaction neurons spike times
    for i in range(NB_CONV_LAYERS):
        figure_params.append(Panel(Conv_i_spikes[i], xlabel="Conv"+str(i+1)+" spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, ConvLayers[i].size)))

    for i in range(NB_CONV_LAYERS):
        figure_params.append(Panel(Conv_i_data[i].filter(name='v')[0], xlabel="Membrane potential (mV) - Conv"+str(i+1)+" layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False))

    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_spikes, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        *figure_params,
        title=title,
        annotations="Simulated with "+ options.simulator.upper()
    ).save(figure_filename)

    visu.print_final_filters()
    print("Figures correctly saved as", figure_filename)
    plt.show()