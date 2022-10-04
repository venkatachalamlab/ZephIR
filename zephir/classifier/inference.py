import copy
from numpy import unravel_index

from .streamers import *
from .model import NeuronClassifier


def perform_inference(
    dataset,
    annotations_df,
    worldlines_df,
    t_idx,
):

    img_shape = (5, 19, 19)
    n_channels_in = 1
    n_channels_out = 7
    n_channels_out2 = 22
    init_nodes = 16
    dev = torch.device('cuda:0')
    state_dict_path = Path(__file__).parent / 'checkpoint.pt'
    state_dict_path2 = Path(__file__).parent / 'checkpoint2.pt'

    # model classifies landmark neurons
    model = NeuronClassifier(
            img_shape=img_shape,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            init_nodes=init_nodes,
            lin_channels_in=7
    ).to(dev)
    checkpoint = torch.load(state_dict_path, map_location=dev)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # model2 classifies all 22 neurons
    model2 = NeuronClassifier(
        img_shape=img_shape,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out2,
        init_nodes=init_nodes,
        lin_channels_in=13
    ).to(dev)
    checkpoint = torch.load(state_dict_path2, map_location=dev)
    state_dict = checkpoint['state_dict']
    model2.load_state_dict(state_dict)

    # reconstruct features here
    input_streamer = InferenceStreamer(
        dataset, annotations_df, worldlines_df, dev
    )
    input_vol, input_lin, wlid_list = input_streamer[t_idx]

    if input_vol.shape[-1] == 0:
        return [], []

    pred_lm = model(input_vol, input_lin)
    pred_lm = pred_lm.detach().cpu().numpy()

    n_neuron = input_vol.shape[0]

    landmark_neuron_id = [0, 0, 0, 0, 0, 0]
    for i in range(n_neuron):
        maxLoc = unravel_index(pred_lm.argmax(), pred_lm.shape)
        if maxLoc[1] != 6:
            # if 10**pred[maxLoc[0]][maxLoc[1]] >= 0.60:
            landmark_neuron_id[maxLoc[1]] = maxLoc[0]
            pred_lm[:, maxLoc[1]] = [float('-inf')] * n_neuron
        pred_lm[maxLoc[0]] = [float('-inf')] * 7

    frame_coordinates = input_lin[:, 1:4]
    landmark_dists = []
    for x, y, z in frame_coordinates:
        for id in landmark_neuron_id:
            if id != -1:
                ax, ay, az = frame_coordinates[id]
                landmark_dists.append(
                    math.sqrt((x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2)
                )

    model2_copy = copy.deepcopy(model2)
    for param in model2_copy.parameters():
        if param.shape == torch.Size([16, 13]):
            idx = []
            for n in range(len(landmark_neuron_id)):
                if landmark_neuron_id[n] == -1:
                    idx.append(n+7)
            _idx = np.setdiff1d(np.arange(13), idx)
            lin_weights = param[:, _idx]

    landmark_dists = np.array(landmark_dists)
    landmark_dists_frame = torch.tensor(
        landmark_dists
    ).view(
        (-1, 6 - landmark_neuron_id.count(-1))
    ).to(torch.float32).to(dev)

    linear_input = torch.cat(
        (input_lin,
         landmark_dists_frame),
        -1)
    # linear_input = input_lin[neuron_counter_2:neuron_counter_2+n_neurons[f], :]
    pred = model2_copy(
        input_vol,
        linear_input, lin_weights
    )
    pred = pred.detach().cpu().numpy()

    output = np.ones(n_neuron) * -1
    for j in range(n_neuron):
        max_loc = unravel_index(pred.argmax(), pred.shape)
        output[max_loc[0]] = max_loc[1]
        pred[:, max_loc[1]] = [float('-inf')] * n_neuron
        pred[max_loc[0]] = [float('-inf')] * 22

    named_output = []
    for i, val in enumerate(output):
        if val == -1:
            named_output.append(f'{wlid_list[i]}'.encode('utf-8'))
        else:
            named_output.append(
                input_streamer.essential_neurons[int(val)]
            )
    print(wlid_list)
    print(named_output)

    return wlid_list, named_output


if __name__ == '__main__':
    dataset = Path('/Users/arvin/Documents/RSI/neuron-classifier/data')
    annotations = get_annotation_df(dataset)
    worldlines = get_worldlines_df(dataset)
    results = perform_inference(dataset, annotations, worldlines, 0)
    print(len(results))
    print(results)
