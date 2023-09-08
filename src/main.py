# ------------------------- Imports
import os
import librosa.display
import librosa.beat
import scipy.stats as st
import warnings
import numpy as np
from scipy.spatial import distance


# ------------------------- Functions
# Ex 2.1
def normalization(toNormalize):
    normalized = np.zeros(toNormalize.shape)
    nl, nc = normalized.shape
    for i in range(nc):
        maxValue = toNormalize[:, i].max()
        minValue = toNormalize[:, i].min()
        normalized[:, i] = (toNormalize[:, i] - minValue) / (maxValue - minValue)

    return normalized


def processFeatures(readFileName, writeFileName):
    # Ex 2.1.1
    top100features = np.genfromtxt(readFileName, delimiter=',')
    nl, nc = top100features.shape
    top100features = top100features[1:, 1:(nc - 1)]

    # Ex 2.1.2
    top100features_normalized = normalization(top100features)

    # Ex 2.1.3
    np.savetxt(writeFileName, top100features_normalized, fmt="%lf", delimiter=',')


# Ex 2.2
def processAudio(readFileName):
    sampleRate = 22050
    useMono = True
    warnings.filterwarnings("ignore")
    F0_minFreq = 20  # 'E0' corresponds to the musical note
    F0_maxFreq = 11025  # 'F9' corresponds to the musical note
    mfcc_dim = 13
    hopLength = 512
    windowLength = 2048  # 92.88 ms

    files = os.listdir(readFileName)
    files.sort()
    numFiles = len(files)

    musicIndex = 0
    all_infos = np.zeros((numFiles, 190))  # 9 features with statistics - (13+1+1+7+1+1+1+1+1)*7 + 1
    features = np.zeros(10)
    features = np.array(features, dtype=object)
    for fileName in files:
        print(fileName, musicIndex)
        fullPath = readFileName + "/" + fileName
        infoFile = librosa.load(fullPath, sr=sampleRate, mono=useMono)[0]

        # mfcc
        features[0] = librosa.feature.mfcc(infoFile, n_mfcc=mfcc_dim)
        # spectral_centroid
        features[1] = librosa.feature.spectral_centroid(infoFile, hop_length=hopLength, win_length=windowLength)
        # spectral_bandwidth
        features[2] = librosa.feature.spectral_bandwidth(infoFile, hop_length=hopLength, win_length=windowLength)
        # spectral_contrast
        features[3] = librosa.feature.spectral_contrast(infoFile, hop_length=hopLength, win_length=windowLength)
        # spectral_flatness
        features[4] = librosa.feature.spectral_flatness(infoFile, hop_length=hopLength, win_length=windowLength)
        # spectral_rollof
        features[5] = librosa.feature.spectral_rolloff(infoFile, hop_length=hopLength, win_length=windowLength)
        # fundamental_frequency
        features[6] = np.array([librosa.yin(infoFile, fmin=F0_minFreq, fmax=F0_maxFreq, hop_length=hopLength)])
        features[6][features[6] == F0_maxFreq] = 0
        # rms
        features[7] = librosa.feature.rms(infoFile, hop_length=hopLength)
        # zero
        features[8] = librosa.feature.zero_crossing_rate(infoFile, hop_length=hopLength)
        # tempo
        features[9] = librosa.beat.tempo(infoFile, hop_length=hopLength)

        counter = 0
        for j in range(features.shape[0] - 1):
            feature = features[j]
            nlFeature = feature.shape[0]
            for k in range(nlFeature):
                all_infos[musicIndex][counter] = np.mean(feature[k, :])
                counter += 1
                all_infos[musicIndex][counter] = np.std(feature[k, :])
                counter += 1
                all_infos[musicIndex][counter] = st.skew(feature[k, :])
                counter += 1
                all_infos[musicIndex][counter] = st.kurtosis(feature[k, :])
                counter += 1
                all_infos[musicIndex][counter] = np.median(feature[k, :])
                counter += 1
                all_infos[musicIndex][counter] = np.max(feature[k, :])
                counter += 1
                all_infos[musicIndex][counter] = np.min(feature[k, :])
                counter += 1
        all_infos[musicIndex][counter] = features[9]
        musicIndex += 1

    return all_infos


def extractFeatures(readFileName, writeFileName):
    all_statistics = processAudio(readFileName)
    statisticsNormalized = normalization(all_statistics)
    np.savetxt(writeFileName, statisticsNormalized, fmt="%lf", delimiter=',')


def getNames(filesNames):
    correctNames = []
    for i in range(6):
        correctNames.append(filesNames[i].split("/")[1].split(".")[0])
    return correctNames


# Ex 3.1 and Ex 3.2
def musicsSimilarity(allMusicsFeaturesFileName, top100FileName, writeFileNames):
    statisticsNormalized = np.genfromtxt(allMusicsFeaturesFileName, delimiter=",")
    top100_N = np.genfromtxt(top100FileName, delimiter=",")
    similarities = np.zeros((6, 900, 900))
    numberOfMusics = statisticsNormalized.shape[0]

    for i in range(numberOfMusics):
        for j in range(i + 1, numberOfMusics):
            # euclidean distance
            # all musics
            statisticsNormalized[i, :] = np.nan_to_num(statisticsNormalized[i, :])
            statisticsNormalized[j, :] = np.nan_to_num(statisticsNormalized[j, :])
            similarities[0][i][j] = distance.euclidean(statisticsNormalized[i, :], statisticsNormalized[j, :])
            similarities[0][j][i] = similarities[0][i][j]
            # top 100 features
            similarities[1][i][j] = distance.euclidean(top100_N[i, :], top100_N[j, :])
            similarities[1][j][i] = similarities[1][i][j]

            # manhattanDistance
            # all musics
            similarities[2][i][j] = distance.cityblock(statisticsNormalized[i, :], statisticsNormalized[j, :])
            similarities[2][j][i] = similarities[2][i][j]
            # top 100 features
            similarities[3][i][j] = distance.cityblock(top100_N[i, :], top100_N[j, :])
            similarities[3][j][i] = similarities[3][i][j]

            # cosine similarity
            # all musics
            similarities[4][i][j] = distance.cosine(statisticsNormalized[i, :], statisticsNormalized[j, :])
            similarities[4][j][i] = similarities[4][i][j]
            # top 100 features
            similarities[5][i][j] = distance.cosine(top100_N[i, :], top100_N[j, :])
            similarities[5][j][i] = similarities[5][i][j]

    for i in range(similarities.shape[0]):
        np.savetxt(writeFileNames[i], similarities[i], fmt="%lf", delimiter=',')


# Ex 3.3
def getQueriesAndSongs(pathQueries, pathMusicFiles):
    queries = []
    songNames = []
    queriesFiles = os.listdir(pathQueries)
    queriesFiles.sort()
    musicFiles = np.genfromtxt(pathMusicFiles, delimiter=',', dtype="str")
    musicFiles = musicFiles[1:, 0]

    for i, q in enumerate(queriesFiles):
        q = q.split('.')[0]
        for j, s in enumerate(musicFiles):
            if i == 0:
                songNames += [s + ".mp3"]
            if q == s:
                queries += [[j, q + ".mp3"]]

    return queries, songNames


def readSimilarities(queries, songNames, readFilesNames, correctNames, show):
    similarities = np.zeros((6, 900, 900))
    for i in range(similarities.shape[0]):
        similarities[i] = np.genfromtxt(readFilesNames[i], delimiter=",")

    recommended = np.zeros((4, 6, 20))  # 4 queries, 6 files, top 20
    for i, q in enumerate(queries):
        for j in range(6):
            values = similarities[j][q[0], :]
            recommended[i][j] = np.array(np.argsort(values)[1:21])

    if show:
        for z, x in enumerate(queries):
            print("Querie:", x[1])
            for y, w in enumerate(recommended[z]):
                print("\n\t", correctNames[y])
                for v in w:
                    print("\t Music:", songNames[int(v)], "| Index:", int(v))
            print()

    return recommended


# Ex 4.1
def metadataSimilarityMatrix(pathMetadataFile, saveFileName):
    metadataRawMatrix = np.genfromtxt(pathMetadataFile, delimiter=',', dtype="str")
    metadata = metadataRawMatrix[1:, [1, 3, 9, 11]]
    metadataSize = metadata.shape[0]
    metadataScores = np.zeros((metadataSize, metadataSize))

    for i in range(metadataSize):  # first song
        for j in range(i + 1):  # second song
            if i == j:
                metadataScores[i][j] = -1
            else:
                score = 0
                for l in range(metadata.shape[1]):  # parameters to compare
                    if l < 2:
                        if metadata[i, l] == metadata[j, l]:
                            score += 1
                    else:
                        listI = metadata[i, l][1:-1].split('; ')
                        listJ = metadata[j, l][1:-1].split('; ')
                        for a in listI:
                            for b in listJ:
                                if a == b:
                                    score += 1
                metadataScores[i][j] = score
                metadataScores[j][i] = score

    np.savetxt(saveFileName, metadataScores, fmt="%lf", delimiter=',')


def getRankingQueries(queries, songNames, pathSimilarityMatrix, show):
    similarityMatrix = np.genfromtxt(pathSimilarityMatrix, delimiter=',')
    rankingQueries = ()

    for q in queries:
        scores = similarityMatrix[q[0]]
        indexRankings = np.flip(np.argsort(scores)[-20:])
        rankingQueries += (indexRankings,)

    if show:
        for z, x in enumerate(queries):
            print("Querie:", x[1])
            for y in rankingQueries[z]:
                print("Music:", songNames[y], "| Index:", y, "| Score:", similarityMatrix[x[0]][y])
            print()

    return rankingQueries


def precision(rankedQueriesMetaData, recommended20, queries, correctNames, show):
    allQueriesPrecision = np.zeros((recommended20.shape[0], recommended20.shape[1]))
    for i in range(recommended20.shape[0]):
        for j in range(recommended20.shape[1]):
            relevant = len(np.intersect1d(rankedQueriesMetaData[i], recommended20[i][j]))
            precisionValue = relevant / 20
            allQueriesPrecision[i][j] = precisionValue

    if show:
        for z, q in enumerate(queries):
            print("Querie:", q[1])
            for i in range(len(allQueriesPrecision[z])):
                print(correctNames[i], "-", allQueriesPrecision[z][i])
            print()


########################################################################################################################

def main():
    # Ex 2.1
    # processFeatures("Features/top100_features.csv", "Features/top100_features_normalized.csv")

    # Ex 2.2
    # extractFeatures("Dataset/all_musics", "Features/features_statistics_normalized.csv")

    # Ex 3.1 and Ex 3.2
    filesNames = ["Features/all_Musics_Euclidean.csv", "Features/top100_Euclidean.csv",
                  "Features/all_Musics_Manhattan.csv", "Features/top100_Manhattan.csv",
                  "Features/all_Musics_Cosine_Similarity.csv", "Features/top100_Cosine_Similarity.csv"]
    correctNames = getNames(filesNames)

    #musicsSimilarity("Features/features_statistics_normalized.csv", "Features/top100_features_normalized.csv", filesNames)

    # Ex 3.3
    queries, songNames = getQueriesAndSongs("Queries", "Dataset/panda_dataset_taffc_annotations.csv")
    # queries = [[10, "MT0000202045.mp3"], [26, "MT0000379144.mp3"], [28, "MT0000414517.mp3"], [59, "MT0000956340.mp3"]]
    recommended = readSimilarities(queries, songNames, filesNames, correctNames, True)

    # Ex 4.1
    # Ex 4.1.2
    metadataSimilarityMatrix("Dataset/panda_dataset_taffc_metadata.csv", "Dataset/metadata_similarity_matrix.csv")
    # Ex 4.1.1
    rankedQueries = getRankingQueries(queries, songNames, "Dataset/metadata_similarity_matrix.csv", True)
    # Ex 4.1.3
    precision(rankedQueries, recommended, queries, correctNames, True)



if __name__ == '__main__':
    main()
