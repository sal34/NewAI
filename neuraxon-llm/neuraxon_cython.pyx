# neuraxon_cython.pyx

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def update_neurons_cython(
    np.ndarray[np.float32_t, ndim=2] neurons,
    np.ndarray[np.float32_t, ndim=1] synaptic_inputs,
    np.ndarray[np.float32_t, ndim=1] modulatory_inputs,
    np.ndarray[np.float32_t, ndim=1] external_inputs,
    np.ndarray[np.float32_t, ndim=1] neuromodulators,
    np.ndarray[np.float32_t, ndim=1] params,
    float dt,
):
    cdef int num_neurons = neurons.shape[0]
    cdef int i
    cdef float total_synaptic, total_modulatory
    cdef float acetylcholine, norepinephrine
    cdef float spontaneous
    cdef float membrane_potential, adaptation, autoreceptor
    cdef float threshold_mod
    cdef float theta_exc, theta_inh
    cdef float activity_level
    cdef float tau_adapt = 100.0
    cdef float tau_auto = 200.0

    for i in range(num_neurons):
        if not neurons[i, 7]:
            continue

        total_synaptic = synaptic_inputs[i]
        total_modulatory = modulatory_inputs[i]

        acetylcholine = neuromodulators[2]
        norepinephrine = neuromodulators[3]

        spontaneous = 0.0
        if np.random.rand() < params[8] * (1 + norepinephrine) * dt:
            spontaneous = np.random.uniform(-0.5, 0.5) * (1 + norepinephrine)

        membrane_potential = neurons[i, 1]
        adaptation = neurons[i, 2]
        autoreceptor = neurons[i, 3]

        membrane_potential += dt / params[4] * (
            -membrane_potential
            + total_synaptic
            + external_inputs[i]
            - adaptation
            + spontaneous
            - autoreceptor
        )

        neurons[i, 1] = membrane_potential

        adaptation += dt / tau_adapt * (
            -adaptation + 0.1 * abs(neurons[i, 0])
        )
        neurons[i, 2] = adaptation

        autoreceptor += dt / tau_auto * (
            -autoreceptor + 0.2 * neurons[i, 0]
        )
        neurons[i, 3] = autoreceptor

        threshold_mod = (acetylcholine - 0.5) * 0.5 - (norepinephrine - 0.5) * 0.2 + total_modulatory * 0.3

        theta_exc = params[5] - threshold_mod - 0.1 * autoreceptor
        theta_inh = params[6] - threshold_mod + 0.1 * autoreceptor

        if membrane_potential > theta_exc:
            neurons[i, 0] = 1
        elif membrane_potential < theta_inh:
            neurons[i, 0] = -1
        else:
            neurons[i, 0] = 0

        activity_level = abs(membrane_potential) / 2.0
        if activity_level < 0.01:
            neurons[i, 4] -= params[9] * dt
        else:
            neurons[i, 4] = min(1.0, neurons[i, 4] + 0.0005 * dt)

        if neurons[i, 6] == 1 and neurons[i, 4] < params[24]:
            if np.random.rand() < 0.001:
                neurons[i, 7] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
def update_synapses_cython(
    np.ndarray[np.float32_t, ndim=2] synapses,
    np.ndarray[np.float32_t, ndim=2] neurons,
    np.ndarray[np.float32_t, ndim=1] neuromodulators,
    np.ndarray[np.float32_t, ndim=1] params,
    float dt,
):
    cdef int num_synapses = synapses.shape[0]
    cdef int i
    cdef int pre_id, post_id
    cdef float pre_state, post_state
    cdef float pre_trace, post_trace
    cdef float tau_trace
    cdef float dopamine, serotonin, acetylcholine
    cdef float delta_w
    cdef float scaling_factor
    cdef float w_fast, w_slow, w_meta
    cdef float integrity

    for i in range(num_synapses):
        pre_id = <int>synapses[i, 0]
        post_id = <int>synapses[i, 1]

        pre_state = neurons[pre_id, 0]
        post_state = neurons[post_id, 0]

        pre_trace = synapses[i, 8]
        post_trace = synapses[i, 9]

        tau_trace = params[20]
        pre_trace += (-pre_trace / tau_trace + (1 if pre_state == 1 else 0)) * dt
        post_trace += (-post_trace / tau_trace + (1 if post_state == 1 else 0)) * dt

        synapses[i, 8] = pre_trace
        synapses[i, 9] = post_trace

        dopamine = neuromodulators[0]
        serotonin = neuromodulators[1]
        acetylcholine = neuromodulators[2]

        delta_w = 0.0
        if pre_state == 1 and post_state == 1:
            delta_w = params[19] * (0.5 + dopamine) * post_trace
        elif pre_state == 1 and post_state == -1:
            delta_w = -params[19] * (0.5 + dopamine) * post_trace
        elif pre_state == -1 and post_state == 1:
            delta_w = -params[19] * 0.1 * post_trace

        scaling_factor = 1.0 + (acetylcholine - 0.5) * 0.2

        w_fast = synapses[i, 2]
        w_slow = synapses[i, 3]
        w_meta = synapses[i, 4]

        w_fast += dt / params[10] * (-w_fast + delta_w * 0.7 * scaling_factor)
        w_fast = max(-1.0, min(1.0, w_fast))
        synapses[i, 2] = w_fast

        w_slow += dt / params[13] * (-w_slow + delta_w * 0.3 * scaling_factor)
        w_slow = max(-1.0, min(1.0, w_slow))
        synapses[i, 3] = w_slow

        w_meta += dt / params[16] * (-w_meta + delta_w * 0.1 * (0.5 + serotonin))
        w_meta = max(-0.5, min(0.5, w_meta))
        synapses[i, 4] = w_meta

        integrity = synapses[i, 7]
        if abs(w_fast) < 0.01 and abs(w_slow) < 0.01:
            integrity -= params[23] * dt
        else:
            integrity = min(1.0, integrity + 0.0001 * dt)
        synapses[i, 7] = integrity

        if synapses[i, 5] and pre_state == 1 and post_state == 1:
            if np.random.rand() < 0.01 * (0.5 + dopamine):
                synapses[i, 5] = 0
