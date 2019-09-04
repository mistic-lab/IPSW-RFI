# Utils for the autoencoder

def round_down(num, divisor):
    return num - (num%divisor)

def get_conditional_indexes(feature, value, comparitor):
    indexes = []
    for i in feature:
        if comparitor(feature, value):
            indexes.append(i)
    return indexes


def build_dataset(directory, indexes, filename, segment_size):
    X = np.array([])
    for i in indexes:

    # load in the waveforms data
        f= directory+filename+'.dat.'+str(i)+'.c64'
        if os.path.exists(f):
            data = np.fromfile(f)
            new_len = round_down(len(data),segment_size)
            if new_len > 0:
                data = data[:new_len]
                X = np.append(X, data)
            else:
                print("Shorty! {} is only {} samples long.".format(filename, data.shape))

    if len(X) % segment_size != 0:
        raise Exception("Array size ({}) is not a multiple of segment size({})".format(len(X),segment_size))
    
    return X
