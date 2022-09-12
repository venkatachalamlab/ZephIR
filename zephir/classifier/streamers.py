import torch
from torch.utils.data import Dataset
from utils import *
import torch.nn as nn
import math


class InferenceStreamer(Dataset):
    def __init__(self, root_path, annotations_df, worldlines_df, dev='cpu'):
        self.dev = dev
        self.volume_streamer = VolumeSelector(root_path)
        self.annotations_df = annotations_df
        self.worldlines_df = worldlines_df

        self.essential_neurons = [
            b'ADFL', b'ADFR',
            b'ADLL', b'ADLR',
            b'ASEL', b'ASER',
            b'ASGL', b'ASGR',
            b'ASHL', b'ASHR',
            b'ASIL', b'ASIR',
            b'ASJL', b'ASJR',
            b'ASKL', b'ASKR',
            b'AWAL', b'AWAR',
            b'AWBL', b'AWBR',
            b'AWCL', b'AWCR'
        ]

    def __len__(self):
        return len(self.volume_streamer)

    def __getitem__(self, idx):
        volume = self.volume_streamer[idx]
        u, annotation = get_annotation(self.annotations_df, idx)

        if len(annotation) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), []

        fullAvgDists = []

        # iterate by frame through data
        # compute coordinates relative to centroid #
        xdata, ydata, zdata = [], [], []
        for neuron in annotation:
            xdata.append(neuron[0])
            ydata.append(neuron[1])
            zdata.append(neuron[2])

        # skip if data is empty
        if len(xdata) == 0:
            return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]),
                    torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

        # compute centroid location
        centroid = (sum(xdata) / len(xdata),
                    sum(ydata) / len(ydata),
                    sum(zdata) / len(zdata))

        neuronDistancesX = []
        neuronDistancesY = []
        neuronDistancesZ = []
        normalizedAnnotation = []
        for neuron in annotation:
            newCoordinates = [neuron[0] - centroid[0], neuron[1] - centroid[1], neuron[2] - centroid[2]]
            normalizedAnnotation.append(newCoordinates)
            neuronDistancesX.append(abs(newCoordinates[0]))
            neuronDistancesY.append(abs(newCoordinates[1]))
            neuronDistancesZ.append(abs(newCoordinates[2]))

        avgNeuronDistancesX = sum(neuronDistancesX) / len(neuronDistancesX)
        avgNeuronDistancesY = sum(neuronDistancesY) / len(neuronDistancesY)
        avgNeuronDistancesZ = sum(neuronDistancesZ) / len(neuronDistancesZ)
        for n in range(len(annotation)):
            normalizedAnnotation[n][0] /= max(0.05, avgNeuronDistancesX)
            normalizedAnnotation[n][1] /= avgNeuronDistancesY
            normalizedAnnotation[n][2] /= avgNeuronDistancesZ

        for n1 in range(len(annotation)):
            _landmarkDists = [-1, -1, -1, -1, -1, -1]
            neuronDists = []
            curLoc = normalizedAnnotation[n1]
            for n2 in range(len(annotation)):
                otherLoc = normalizedAnnotation[n2]
                dist = math.sqrt((curLoc[0]-otherLoc[0])**2+(curLoc[1]-otherLoc[1])**2+(curLoc[2]-otherLoc[2])**2)
                neuronDists.append(dist)
            fullAvgDists += sorted(neuronDists)[1:4]

        croppedVolumes = []
        for neuron in annotation:
            z = int((neuron[2] + 1) * volume.shape[1] / 2)
            y = int((neuron[1] + 1) * volume.shape[2] / 2)
            x = int((neuron[0] + 1) * volume.shape[3] / 2)
            padding = nn.ConstantPad3d((13, 13, 13, 13, 4, 4), 0)
            paddedVol = padding(volume)
            croppedVolume = paddedVol[:, z+2:z+7, y+4:y+23, x+4:x+23]
            croppedVolume = torch.div(croppedVolume, np.mean(volume.numpy()))
            croppedVolume = torch.div(croppedVolume, np.max(croppedVolume.numpy()))
            if croppedVolume.shape != (3, 5, 19, 19):
                print(x,y,z)
                print(volume.shape)
                print(paddedVol.shape)
                print(croppedVolume.shape)

            # need to detect when indexing goes below 0 or over max size
            # need to handle improper indexing cases (pad original volume?)
            croppedVolumes.append(np.array(croppedVolume))
        croppedVolumes = np.array(croppedVolumes)
        croppedVolumes = croppedVolumes / np.max(croppedVolumes)

        # croppedVolumes = np.where(np.array(croppedVolumes) > 0.05, 1, 0)
        croppedVolumes = torch.tensor(np.array(croppedVolumes))

        volumesReshape = croppedVolumes[:, 1, ...].view((-1, *croppedVolumes.shape[3:6]))
        volumesReshape = volumesReshape.to(torch.float32)
        normalizedAnnotation = torch.tensor(normalizedAnnotation).view((-1, 3))
        normalizedAnnotation = normalizedAnnotation.to(torch.float32)


        brightness_list = []
        for volume in volumesReshape:
            miniVol = volume[((volume.shape[0] // 2) - 1):((volume.shape[0] // 2) + 1),
                             ((volume.shape[1] // 2) - 3):((volume.shape[1] // 2) + 3),
                             ((volume.shape[2] // 2) - 3):((volume.shape[2] // 2) + 3)]
            brightness = np.mean(miniVol.numpy())*100
            brightness_list.append(brightness)
        avgBrightness = sum(brightness_list) / len(brightness_list)

        for n in range(len(brightness_list)):
            if avgBrightness != 0:
                brightness_list[n] /= avgBrightness

        brightnessList = torch.tensor(np.array(brightness_list))
        brightnessList = brightnessList.view(volumesReshape.shape[0], 1)
        brightnessList = brightnessList.to(torch.float32)

        fullAvgDists = torch.tensor(np.array(fullAvgDists))
        fullAvgDists = fullAvgDists.view(volumesReshape.shape[0], 3)
        fullAvgDists = fullAvgDists.to(torch.float32)

        coordinates = normalizedAnnotation.view((-1, 3))
        brightness = brightnessList.view((-1, 1))
        coordinates = torch.cat((coordinates, brightness), -1)
        avgDists = fullAvgDists.view((-1, 3))
        coordinates = torch.cat((coordinates, avgDists), -1)

        # volumesReshape.shape = (N*T, C, Z, Y, X)
        return (
            volumesReshape.to(self.dev).to(torch.float32),
            coordinates.to(self.dev).view(-1, 7).to(torch.float32),
            u.to(self.dev).to(torch.float32)
        )


class VolumeSelector(Dataset):
    def __init__(self, root_path):
        self.root_path = Path(root_path)

        n_datasets = 0
        for f in self.root_path.iterdir():
            if not str(f.parts[-1])[0] == '.':
                n_datasets += 1
        self.n_datasets = n_datasets

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        p = self.root_path
        vol = get_slice(p, idx)
        vol = torch.tensor(vol)
        return vol


class TrainingStreamer(Dataset):
    def __init__(self, root_path, sample_size=10, dev='cpu'):
        self.dev = dev
        self.volume_streamer = VolumeStreamer(root_path, sample_size)
        self.worldline_streamer = WorldlineStreamer(root_path)
        self.annotation_streamer = AnnotationStreamer(root_path)

    def __len__(self):
        return len(self.volume_streamer)

    def __getitem__(self, idx):

        annotation_df = self.annotation_streamer[idx]
        times, volumes = self.volume_streamer[idx]
        worldline_ids, names = self.worldline_streamer[idx]
        names = names[None, ...].expand((len(times), -1))
        # names = names.reshape((-1))

        # filtering annotation by worldline_id
        annotations = []
        for t in times:
            u, _annot = get_annotation(annotation_df, t)
            u_idx = np.array([np.where(u == w)[0][-1]
                              for w in worldline_ids if w in u], dtype=int)
            _annot = _annot[u_idx, ...]
            annotations.append(_annot)

        centroid = (0, 0, 0)
        normalizedAnnotations = []
        fullAvgDists = []
        fullLandmarkDists = []
        numNeurons = []
        neuronDistsAvg = []

        # iterate by frame through data
        for t, frame in enumerate(annotations):
            # compute coordinates relative to centroid #
            xdata, ydata, zdata = [], [], []
            numNeurons.append(len(frame))
            for neuron in frame:
                xdata.append(neuron[0])
                ydata.append(neuron[1])
                zdata.append(neuron[2])

            # skip if data is empty
            if len(xdata) == 0:
                return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]),
                        torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

            # compute centroid location
            centroid = (sum(xdata) / len(xdata), sum(ydata) / len(ydata), sum(zdata) / len(zdata))

            neuronDistancesX = []
            neuronDistancesY = []
            neuronDistancesZ = []
            normalizedAnnotation = []
            for neuron in frame:
                newCoordinates = [neuron[0] - centroid[0], neuron[1] - centroid[1], neuron[2] - centroid[2]]
                normalizedAnnotation.append(newCoordinates)
                neuronDistancesX.append(abs(newCoordinates[0]))
                neuronDistancesY.append(abs(newCoordinates[1]))
                neuronDistancesZ.append(abs(newCoordinates[2]))
            normalizedAnnotations.append(normalizedAnnotation)

            avgNeuronDistancesX = sum(neuronDistancesX) / len(neuronDistancesX)
            avgNeuronDistancesY = sum(neuronDistancesY) / len(neuronDistancesY)
            avgNeuronDistancesZ = sum(neuronDistancesZ) / len(neuronDistancesZ)
            for n in range(len(frame)):
                normalizedAnnotations[t][n][0] /= max(0.05, avgNeuronDistancesX)
                normalizedAnnotations[t][n][1] /= avgNeuronDistancesY
                normalizedAnnotations[t][n][2] /= avgNeuronDistancesZ

            # avgDists = []
            topNeuronDists = []
            landmarkDists = []
            for n1 in range(len(frame)):
                _landmarkDists = [-1, -1, -1, -1, -1, -1]
                neuronDists = []
                curLoc = normalizedAnnotations[t][n1]
                for n2 in range(len(frame)):
                    otherLoc = normalizedAnnotations[t][n2]
                    dist = math.sqrt((curLoc[0]-otherLoc[0])**2+(curLoc[1]-otherLoc[1])**2+(curLoc[2]-otherLoc[2])**2)
                    neuronDists.append(dist)
                for i in range(len(neuronDists)):
                    if names[t][i] in [2, 3, 10, 11, 14, 15]:
                        _landmarkDists[[2, 3, 10, 11, 14, 15].index(names[t][i])] = neuronDists[i]
                if -1 in _landmarkDists:
                    return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]),
                            torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))
                landmarkDists.append(_landmarkDists)
                topNeuronDists += sorted(neuronDists)[1:4]
            fullAvgDists += topNeuronDists
            fullLandmarkDists += landmarkDists
        volumes = torch.stack(volumes, dim=0)

        croppedVolumes = []
        for t in range(volumes.shape[0]):
            vol = volumes[t]
            frame = annotations[t]
            croppedVolumesPerFrame = []
            for neuron in frame:
                z = int((neuron[2] + 1) * vol.shape[1] / 2)
                y = int((neuron[1] + 1) * vol.shape[2] / 2)
                x = int((neuron[0] + 1) * vol.shape[3] / 2)
                padding = nn.ConstantPad3d((13, 13, 13, 13, 4, 4), 0)
                paddedVol = padding(vol)
                croppedVolume = paddedVol[:, z+2:z+7, y+4:y+23, x+4:x+23]
                croppedVolume = torch.div(croppedVolume, np.mean(vol.numpy()))
                croppedVolume = torch.div(croppedVolume, np.max(croppedVolume.numpy()))
                if croppedVolume.shape != (3, 5, 19, 19):
                    print(x,y,z)
                    print(vol.shape)
                    print(paddedVol.shape)
                    print(croppedVolume.shape)

                # need to detect when indexing goes below 0 or over max size
                # need to handle improper indexing cases (pad original volume?)
                croppedVolumesPerFrame.append(np.array(croppedVolume))
            croppedVolumesPerFrame = np.array(croppedVolumesPerFrame)
            croppedVolumesPerFrame = croppedVolumesPerFrame / np.max(croppedVolumesPerFrame)
            croppedVolumes.append(croppedVolumesPerFrame)

        # croppedVolumes = np.where(np.array(croppedVolumes) > 0.05, 1, 0)
        croppedVolumes = torch.tensor(np.array(croppedVolumes))

        volumesReshape = croppedVolumes[:, :, 1, ...].view((-1, 1, *croppedVolumes.shape[3:6]))
        volumesReshape = volumesReshape.to(torch.float32)
        normalizedAnnotations = torch.tensor(normalizedAnnotations).view((-1, 3))
        normalizedAnnotations = normalizedAnnotations.to(torch.float32)
        names = names.type(torch.LongTensor)
        namesLandmark = torch.clone(names)
        nameMap = [2, 3, 10, 11, 14, 15]
        for f in range(len(names)):
            for n in range(len(names[f])):
                name = int(names[f][n])
                if name in nameMap:
                    namesLandmark[f][n] = nameMap.index(name)
                else:
                    namesLandmark[f][n] = 6

        brightnessList = []
        for t in volumesReshape:
            frameBrightness = []
            for volume in t:
                miniVol = volume[((volume.shape[0] // 2) - 1):((volume.shape[0] // 2) + 1),
                          ((volume.shape[1] // 2) - 3):((volume.shape[1] // 2) + 3),
                          ((volume.shape[2] // 2) - 3):((volume.shape[2] // 2) + 3)]
                brightness = np.mean(miniVol.numpy())*100
                frameBrightness.append(brightness)
            avgBrightness = sum(frameBrightness) / len(frameBrightness)
            for n in range(len(frameBrightness)):
                if avgBrightness != 0:
                    frameBrightness[n] /= avgBrightness
            brightnessList.append(frameBrightness)
        brightnessList = torch.tensor(np.array(brightnessList))
        brightnessList = brightnessList.view(volumesReshape.shape[0], 1)
        brightnessList = brightnessList.to(torch.float32)
        fullAvgDists = torch.tensor(np.array(fullAvgDists))
        fullAvgDists = fullAvgDists.view(volumesReshape.shape[0], 3)
        fullAvgDists = fullAvgDists.to(torch.float32)
        numNeurons = torch.tensor(np.array(numNeurons))

        fullLandmarkDists = torch.tensor(np.array(fullLandmarkDists))
        fullLandmarkDists = fullLandmarkDists.to(torch.float32)
        fullLandmarkDists = fullLandmarkDists.view(volumesReshape.shape[0], 6)


        # volumesReshape.shape = (N*T, C, Z, Y, X)
        return (volumesReshape.to(self.dev), normalizedAnnotations.to(self.dev), brightnessList.to(self.dev),
                fullAvgDists.to(self.dev), numNeurons.to(self.dev), fullLandmarkDists.to(self.dev),
                names.to(self.dev), namesLandmark.to(self.dev))


class VolumeStreamer(Dataset):
    def __init__(self, root_path, sample_size=10):
        self.root_path = Path(root_path)
        self.sample_size = sample_size

        n_datasets = 0
        for f in self.root_path.iterdir():
            if not str(f.parts[-1])[0] == '.':
                n_datasets += 1
        self.n_datasets = n_datasets

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        p = self.root_path / str(idx)
        all_times = get_times(p)
        times_to_sample = np.random.randint(len(all_times), size=self.sample_size)
        sampled_volumes = []
        for t in times_to_sample:
            vol = get_slice(p, t)
            vol = torch.tensor(vol)
            sampled_volumes.append(vol)

        return times_to_sample, sampled_volumes


class WorldlineStreamer(Dataset):
    def __init__(self, root_path):
        self.root_path = Path(root_path)

        n_datasets = 0
        for f in self.root_path.iterdir():
            if not str(f.parts[-1])[0] == '.':
                n_datasets += 1
        self.n_datasets = n_datasets

        # see choosing_neurons.ipynb
        self.essential_neurons = [
            b'ADFL', b'ADFR',
            b'ADLL', b'ADLR',
            b'ASEL', b'ASER',
            b'ASGL', b'ASGR',
            b'ASHL', b'ASHR',
            b'ASIL', b'ASIR',
            b'ASJL', b'ASJR',
            b'ASKL', b'ASKR',
            b'AWAL', b'AWAR',
            b'AWBL', b'AWBR',
            b'AWCL', b'AWCR'
        ]

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        p = self.root_path / str(idx)
        worldlines = get_worldlines_df(p)

        # filtering by name length
        filter = (worldlines["name"].map(len) > 2)
        names = worldlines["name"][filter]

        # filtering worldlines with names in the essential neurons set only
        ids = worldlines["id"][filter][[n in self.essential_neurons for n in names]]
        names = names[[n in self.essential_neurons for n in names]]

        # names_vector = np.zeros((len(names), len(self.essential_neurons)))
        # names_vector[
        #     range(len(names)),
        #     [self.essential_neurons.index(n) for n in names]] = 1
        names_vector = np.array([self.essential_neurons.index(n) for n in names])

        return list(ids), torch.tensor(names_vector)


class AnnotationStreamer(Dataset):
    def __init__(self, root_path):
        self.root_path = Path(root_path)

        n_datasets = 0
        for f in self.root_path.iterdir():
            if not str(f.parts[-1])[0] == '.':
                n_datasets += 1
        self.n_datasets = n_datasets

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        p = self.root_path / str(idx)
        annotations = get_annotation_df(p)
        return annotations
