import copy
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .streamers import *
from .model import NeuronClassifier


def main(
        root_path,
        img_shape=(5, 64, 64),
        n_channels_in=3,
        n_channels_out=1,
        n_channels_out2=1,
        init_nodes=16,
        dev=torch.device('cpu'),
        n_epoch=10,
        lr_init=0.1,
        batch_size=1,
        validation_split=0.,
        state_dict_path=None,
        state_dict_path2=None
):
    print('\n\nCompiling model...')

    # model classifies landmark neurons
    model = NeuronClassifier(
        img_shape=img_shape,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        init_nodes=init_nodes,
        lin_channels_in=7
    ).to(dev)

    if state_dict_path is not None:
        print('Loading model weights from previous checkpoint...')
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

    if state_dict_path2 is not None:
        print('Loading model weights from previous checkpoint...')
        checkpoint = torch.load(state_dict_path2, map_location=dev)
        state_dict = checkpoint['state_dict']
        model2.load_state_dict(state_dict)

    # defining loss and optimizer functions
    # weight decay to prevent overfitting
    loss_function = nn.NLLLoss()
    optimizer = optim.Adadelta(model2.parameters(), lr=lr_init, weight_decay=0.001)

    print('\n\nCompiling data loaders...')
    training_streamer = TrainingStreamer(root_path, sample_size=10, dev=dev)

    # splitting data to withhold some fraction for training validation
    n_data = len(training_streamer)
    n_val = int(np.floor(validation_split * n_data))
    n_train = n_data - n_val
    if n_val > 0:
        # training_streamer, stream_val = random_split(training_streamer, [n_train, n_val])
        stream_val = Subset(training_streamer, range(n_val))
        training_streamer = Subset(training_streamer, range(n_val, n_data))
        validation_input = DataLoader(
            stream_val,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    training_input = DataLoader(
        training_streamer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    trainLosses = []
    valLosses = []
    last_loss = 100
    earlyStoppingCount = 0
    patience = 6
    # training loop
    model2.train()
    pbar = tqdm(range(n_epoch), desc='Training', unit='epochs')
    for epoch in pbar:
        loss_list = []
        bbar = tqdm(training_input, desc='Fitting batches', unit='batch', leave=False)

        for i, batch in enumerate(bbar):
            optimizer.zero_grad()

            # get data from batch
            volumes, coordinates, brightness, avgDists, numNeurons, landmarkDists, names, namesLandmark = batch

            # skip if data is empty
            with torch.no_grad():
                if volumes.shape[-1] == 0:
                    continue

            # reshape and concatenate data to input into model
            volumes = volumes.view((-1, *volumes.shape[2:6]))
            coordinates = coordinates.view((-1, 3))
            brightness = brightness.view((-1, 1))
            coordinates = torch.cat((coordinates, brightness), -1)
            avgDists = avgDists.view((-1, 3))
            coordinates = torch.cat((coordinates, avgDists), -1)
            landmarkDists = landmarkDists.view((-1, 6))
            coordinates = torch.cat((coordinates, landmarkDists), -1)

            # used model to output prediction (log probability distribution) and compute loss
            pred = model2(volumes, coordinates)
            names = names.view((-1))
            loss = loss_function(pred, names)

            # backpropagation - update weights
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_list.append(loss.clone().detach())
                bbar.set_postfix(Loss=f'{loss.item():.5f}')

        with torch.no_grad():
            mean_loss = torch.mean(torch.tensor(loss_list, device=dev))
            trainLosses.append(mean_loss.item())
            f = open(f'{root_path.parent.as_posix()}/arvind_tests/trainLosses.txt', "a")
            f.write(f"{mean_loss.item()}")
            f.close()

            # validation on withheld data
            val_loss = torch.tensor([0], device=dev)
            if n_val > 0:
                model2.eval()
                vbar = tqdm(validation_input, desc='Validating batches', unit='batch', leave=False)
                for val_batch in vbar:
                    # same process to compute validation loss
                    volumes, coordinates, brightness, avgDists, numNeurons, landmarkDists, names, namesLandmark = val_batch
                    if volumes.shape[-1] == 0:
                        continue

                    volumes = volumes.view((-1, *volumes.shape[2:6]))
                    coordinates = coordinates.view((-1, 3))
                    brightness = brightness.view((-1, 1))
                    coordinates = torch.cat((coordinates, brightness), -1)
                    avgDists = avgDists.view((-1, 3))
                    coordinates = torch.cat((coordinates, avgDists), -1)
                    landmarkDists = landmarkDists.view((-1, 6))
                    coordinates = torch.cat((coordinates, landmarkDists), -1)
                    val_pred = model2(volumes, coordinates)

                    _val_loss = loss_function(val_pred, names.view((-1)))
                    val_loss = val_loss + _val_loss / n_val
                valLosses.append(val_loss.item())
                f = open(f'{root_path.parent.as_posix()}/arvind_tests/valLosses.txt', "a")
                f.write(f"{val_loss.item()}")
                f.close()

            pbar.set_postfix(Loss=f'{mean_loss.item():.5f}',
                             Val_Loss=f'{val_loss.item():.5f}')

            checkpoint = {
                'state_dict': model2.state_dict(),
            }
            torch.save(checkpoint, Path(__file__).parent / 'checkpoint2.pt')

            # early stopping to prevent overfitting
            if val_loss > (last_loss+0.2):
                earlyStoppingCount += 1
            if earlyStoppingCount >= patience:
                break
            last_loss = val_loss

    ##############################################################################
    # training evaluation on validation data
    ##############################################################################
    model2.eval()
    with torch.no_grad():
        numCorrect = 0
        numTotal = 0
        if n_val > 0:
            vbar = tqdm(validation_input, desc='Validating batches', unit='batch', leave=False)
            confusionMatrix = np.zeros((22, 22))
            for val_batch in vbar:
                volumes, coordinates, brightness, avgDists, numNeurons, landmarkDists, names, namesLandmark = val_batch
                if volumes.shape[-1] == 0:
                    continue
                volumes = volumes.view((-1, *volumes.shape[2:6]))
                coordinates = coordinates.view((-1, 3))
                brightness = brightness.view((-1, 1))
                linearData = torch.cat((coordinates, brightness), -1)
                avgDists = avgDists.view((-1, 3))
                linearData = torch.cat((linearData, avgDists), -1)
                numNeurons = numNeurons.view((-1, *numNeurons.shape[2:]))
                val_pred = model(volumes, linearData)
                val_pred = val_pred.detach().cpu().numpy()

                namesLandmark = namesLandmark.view((-1))
                names = names.view((-1))
                landmarkDists = landmarkDists.view((-1, 6))

                neuronCounter = 0
                frameCounter = 0
                predsAndTruths = []
                fullLandmarkDists = []
                landmarkNeuronIDs = []

                # for i in range(len(val_pred)+1):
                #     landmarkDists = []
                #     if neuronCounter == numNeurons[frameCounter]:
                #         pred = val_pred[i - neuronCounter:i]
                #         landmarkNeuronID = [-1,-1,-1,-1,-1,-1,-1]
                #         for j in range(numNeurons[frameCounter]):
                #             maxLoc = unravel_index(pred.argmax(), pred.shape)
                #             predsAndTruths.append((maxLoc[1], namesLandmark[i-neuronCounter+maxLoc[0]]))
                #             x = [float('-inf')]*7
                #             y = [float('-inf')]*neuronCounter
                #
                #             # pred[:, maxLoc[1]] = y
                #
                #             if maxLoc[1] != 6:
                #                 landmarkNeuronID[maxLoc[1]] = maxLoc[0]
                #                 pred[:, maxLoc[1]] = y
                #             pred[maxLoc[0]] = x
                #         # print(landmarkNeuronID[0:6])
                #         landmarkNeuronIDs.append(landmarkNeuronID[0:6])
                #         frameCoordinates = coordinates[i-neuronCounter:i]
                #         # print(len(frameCoordinates))
                #         for x, y, z in frameCoordinates:
                #             for id in landmarkNeuronID[0:6]:
                #                 if id != -1:
                #                     ax, ay, az = frameCoordinates[id]
                #                     landmarkDists.append(math.sqrt(((x-ax)**2)+((y-ay)**2)+((z-az)**2)))
                #         fullLandmarkDists.append(landmarkDists)
                #
                #         neuronCounter = 0
                #         frameCounter += 1
                #     neuronCounter += 1

                for f in range(10):
                    pred = val_pred[neuronCounter:neuronCounter+numNeurons[f]]
                    landmarkNeuronID = [0, 0, 0, 0, 0, 0]
                    for i in range(numNeurons[f]):
                        maxLoc = np.unravel_index(pred.argmax(), pred.shape)
                        predsAndTruths.append((maxLoc[1], namesLandmark[neuronCounter+maxLoc[0]]))
                        x = [float('-inf')]*7
                        y = [float('-inf')]*numNeurons[f]

                        # pred[:, maxLoc[1]] = y

                        if maxLoc[1] != 6:
                            # if 10**pred[maxLoc[0]][maxLoc[1]] >= 0.60:
                            landmarkNeuronID[maxLoc[1]] = maxLoc[0]
                            pred[:, maxLoc[1]] = y
                        pred[maxLoc[0]] = x
                    landmarkNeuronIDs.append(landmarkNeuronID)
                    frameCoordinates = coordinates[neuronCounter:neuronCounter+numNeurons[f]]
                    landmarkDists = []
                    for x, y, z in frameCoordinates:
                        for id in landmarkNeuronID:
                            if id != -1:
                                ax, ay, az = frameCoordinates[id]
                                landmarkDists.append(math.sqrt(((x-ax)**2)+((y-ay)**2)+((z-az)**2)))
                    fullLandmarkDists.append(landmarkDists)
                    neuronCounter += numNeurons[f]

                neuronCounter2 = 0
                val_preds = []
                for f in range(len(landmarkNeuronIDs)):
                    model2Copy = copy.deepcopy(model2)
                    for param in model2Copy.parameters():
                        if param.shape == torch.Size([16, 13]):
                            idx = []
                            for n in range(len(landmarkNeuronIDs[f])):
                                if landmarkNeuronIDs[f][n] == -1:
                                    idx.append(n+7)
                            _idx = np.setdiff1d(np.arange(13), idx)
                            linWeights = param[:, _idx]
                    fullLandmarkDists[f] = np.array(fullLandmarkDists[f])
                    landmarkDistsFrame = torch.tensor(np.array(fullLandmarkDists[f]))
                    landmarkDistsFrame = landmarkDistsFrame.view((-1, 6-landmarkNeuronIDs[f].count(-1)))
                    landmarkDistsFrame = landmarkDistsFrame.to(torch.float32)
                    landmarkDistsFrame = landmarkDistsFrame.to(dev)

                    linearInput = torch.cat((linearData[neuronCounter2:neuronCounter2+numNeurons[f]], landmarkDistsFrame), -1)
                    names = names.view((-1))
                    #print(volumes[neuronCounter2:neuronCounter2+numNeurons[f]].shape, linearInput.shape)


                    val_pred2 = model2Copy(volumes[neuronCounter2:neuronCounter2+numNeurons[f]],
                                           linearInput, linWeights)
                    val_pred2 = val_pred2.detach().cpu().numpy()

                    val_preds.append(val_pred2)
                    neuronCounter2 += numNeurons[f]
                val_preds = np.array(val_preds)
                val_preds = val_preds.flatten()
                val_preds = torch.tensor(val_preds).view(-1, 22)

                # landmarkDists = landmarkDists.view((-1, 6))
                # linearData = torch.cat((linearData, landmarkDists), -1)
                # val_preds = model2(volumes, linearData)


                neuronCounter = 0
                predsAndTruths = []
                for f in range(len(numNeurons)):
                    pred = val_preds[neuronCounter:neuronCounter+numNeurons[f]].cpu().numpy()
                    for j in range(numNeurons[f]):
                        maxLoc = np.unravel_index(pred.argmax(), pred.shape)
                        predsAndTruths.append((maxLoc[1], names[neuronCounter+maxLoc[0]]))
                        x = [float('-inf')]*22
                        y = [float('-inf')]*numNeurons[f]
                        pred[:, maxLoc[1]] = y
                        pred[maxLoc[0]] = x
                    neuronCounter += numNeurons[f]

                for pred, truth in predsAndTruths:
                    if pred == truth:
                        numCorrect += 1
                    numTotal += 1
                    confusionMatrix[truth][pred] += 1

                # for i in range(len(val_preds)):
                #         pred = np.where(val_preds[i] == max(val_preds[i]))[0][0]
                #         if pred == names[i]:
                #             numCorrect += 1
                #         numTotal += 1
                #         confusionMatrix[pred][names[i]] += 1
        else:
            val_batch = next(iter(training_input))
            volumes, coordinates, brightness, avgDists, numNeurons, landmarkDists, names, namesLandmark = val_batch
            volumes = volumes.view((-1, *volumes.shape[2:6]))
            coordinates = coordinates.view((-1, 3))
            brightness = brightness.view((-1, 1))
            coordinates = torch.cat((coordinates, brightness), -1)
            avgDists = avgDists.view((-1, 3))
            coordinates = torch.cat((coordinates, avgDists), -1)
            val_pred = model(volumes, coordinates)
            val_pred = val_pred.detach().cpu().numpy()

            names = names.view((-1))
            confusionMatrix = np.zeros((22, 22))
            for i in range(len(val_pred)):
                pred = np.where(val_pred[i] == max(val_pred[i]))[0][0]
                if pred == names[i]:
                    numCorrect += 1
                numTotal += 1
                confusionMatrix[names[i]][pred] += 1

    essential_neurons = [
        'ADFL', 'ADFR',
        'ADLL', 'ADLR',
        'ASEL', 'ASER',
        'ASGL', 'ASGR',
        'ASHL', 'ASHR',
        'ASIL', 'ASIR',
        'ASJL', 'ASJR',
        'ASKL', 'ASKR',
        'AWAL', 'AWAR',
        'AWBL', 'AWBR',
        'AWCL', 'AWCR'
    ]
    # essential_neurons = [
    #     'ADLL', 'ADLR',
    #     'ASIL', 'ASIR',
    #     'ASKL', 'ASKR',
    #     'None'
    # ]

    # display accuracy and save important plots (confusion matrix, validation loss, training loss)
    print(f'{numCorrect} / {numTotal} correct ({(numCorrect / numTotal)*100:.2f}%)')

    cm = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=essential_neurons)
    cm.plot()
    plt.savefig(f'{root_path.parent.as_posix()}/arvind_tests/confusionMatrix.png')

    if n_epoch > 0:
        plt.figure(figsize=(10, 5))
        plt.title("Validation Loss")
        plt.plot(valLosses, label="val")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{root_path.parent.as_posix()}/arvind_tests/valloss.png')

        plt.figure(figsize=(10, 5))
        plt.title("Training")
        plt.plot(trainLosses, label="train")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{root_path.parent.as_posix()}/arvind_tests/trainloss.png')

    # save visual display of model weights for convolutional model
    # transposedWeights = torch.transpose(model.output_layer[0].weight, 0, 1)
    # averagedWeights = []
    #
    # output_img_shape = model.output_img_shape
    # # print(np.max(transposedWeights.detach().numpy()))
    # for col in transposedWeights:
    #     averagedWeights.append((sum(col) / len(col)).item())
    # reformattedWeights = torch.tensor(averagedWeights[0:len(averagedWeights) - (init_nodes * 3)])
    # reformattedWeights = reformattedWeights.view(*output_img_shape, init_nodes * 4)
    # means = []
    # for a in reformattedWeights:
    #     for b in a:
    #         for weight in b:
    #             means.append(weight.mean().item())
    # means = torch.tensor(means).view(output_img_shape)
    #
    # output = np.max(means.numpy(), axis=0)
    # plt.figure()
    # if np.max(output) != 0:
    #     output = output / np.max(output)
    # plt.imshow(output, interpolation='nearest')
    # plt.savefig(f'{root_path.parent.as_posix()}/arvind_tests/weights.png')
    return


if __name__ == '__main__':
    main(
        root_path=Path('/Users/arvin/Documents/RSI/neuron-classifier/data'), #'W:\\Jin\\ZM10104\\data'
        img_shape=(5, 19, 19),
        n_channels_in=1,
        n_channels_out=7,
        n_channels_out2=22,
        init_nodes=16,
        dev=torch.device('cuda:0'),
        n_epoch=0,
        lr_init=1.0,
        batch_size=1,
        validation_split=0.1,
        state_dict_path=Path(__file__).parent / 'checkpoint.pt',
        state_dict_path2=Path(__file__).parent / 'checkpoint2.pt'
    )
