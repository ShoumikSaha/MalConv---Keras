import os
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
from sklearn.neighbors import NearestNeighbors

import utils
from preprocess import preprocess

parser = argparse.ArgumentParser(description='Malconv-keras classifier training')
parser.add_argument('--save_path', type=str, default='../saved/adversarial_samples',    help="Directory for saving adv samples")
parser.add_argument('--model_path', type=str, default='../saved/malconv.h5',            help='Path to target model')
parser.add_argument('--log_path', type=str, default='../saved/adversarial_log.csv',     help="[csv file] Adv sample generation log")
parser.add_argument('--pad_percent', type=float, default=0.1,                        help="padding percentage to origin file")
parser.add_argument('--targetClass', type=int, default=1,                            help="target class. default:1")
parser.add_argument('--step_size', type=float, default=0.01,                         help="optimiztion step size for fgsm, senitive")
parser.add_argument('--limit', type=float, default = 0.,                             help="limit gpu memory percentage")
parser.add_argument('csv', type=str,                                                 help="[csv file] Filenames")

def fgsm(model, inp, pad_idx, pad_len, e, step_size=0.001, target_class=0):
    adv = inp.copy()
    loss = K.mean(model.output[:, target_class])
    grads = K.gradients(loss, model.layers[1].output)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-8)

    mask = np.zeros(model.layers[1].output.shape[1:]) # embedding layer output shape
    mask[pad_idx:pad_idx+pad_len] = 1
    grads *= K.constant(mask)

    iterate = K.function([model.layers[1].output], [loss, grads])
    g = 0.
    step = int(1/step_size)*10
    for _ in range(step):
        loss_value, grads_value = iterate([adv])
        grads_value *= step_size
        g += grads_value
        adv += grads_value
        #print (e, loss_value, grads_value.mean(), end='\r')
        if loss_value >= 0.9:
            break

    return adv, g, loss_value


def gen_adv_samples(model, fn_list, pad_percent=0.1, step_size=0.001, target_class=1):

    ###   search for nearest neighbor in embedding space ###
    def emb_search(org, adv, pad_idx, pad_len, neigh):
        out = org.copy()
        for idx in range(pad_idx, pad_idx+pad_len):
            target = adv[idx].reshape(1, -1)
            best_idx = neigh.kneighbors(target, 1, False)[0][0]
            out[0][idx] = best_idx
        return out


    max_len = int(model.input.shape[1])
    emb_layer = model.layers[1]
    emb_weight = emb_layer.get_weights()[0]
    print(max_len, len(emb_weight), len(emb_weight[0]))
    inp2emb = K.function([model.input] , [emb_layer.output]) # [function] Map sequence to embedding

    # Build neighbor searches
    neigh = NearestNeighbors(n_neighbors=1).fit(emb_weight)
    #neigh.fit(emb_weight)

    log = utils.logger()
    adv_samples = []

    for e, fn in enumerate(fn_list):

        ###   run one file at a time due to different padding length, [slow]
        inp, len_list = preprocess([fn], max_len)
        inp_emb = np.squeeze(np.array(inp2emb([inp])), 0)

        pad_idx = len_list[0]
        pad_len = max(min(int(len_list[0]*pad_percent), max_len-pad_idx), 0)
        org_pred = np.argmax(model.predict(inp)[0])    ### origianl score, 0 -> malicious, 1 -> benign
        loss, pred = float('nan'), float('nan')

        final_adv = np.zeros(max_len, dtype=int)

        if pad_len > 0:

            if np.argmax(org_pred) != target_class:
                adv_emb, gradient, loss = fgsm(model, inp_emb, pad_idx, pad_len, e, step_size, target_class)
                adv = emb_search(inp, adv_emb[0], pad_idx, pad_len, neigh)
                pred = np.argmax(model.predict(adv)[0]).astype(float)
                final_adv = adv[0][:pad_idx+pad_len]

            else: # use origin file
                final_adv = inp[0][:pad_idx]


        log.write(fn, org_pred, pad_idx, pad_len, loss, pred)

        # sequence to bytes
        bin_adv = bytes(list(final_adv))
        adv_samples.append(bin_adv)

    return adv_samples, log




if __name__ == '__main__':
    args = parser.parse_args()

    # limit gpu memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)

    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    model = load_model(args.model_path)
    print(model.summary())

    adv_samples, log = gen_adv_samples(model, fn_list, args.pad_percent, args.step_size, args.targetClass)

    # write to file
    log.save(args.log_path)
    for fn, adv in zip(fn_list, adv_samples):
        _fn = fn.split('/')[-1]
        dst = os.path.join(args.save_path, _fn)
        with open(dst, 'wb') as f:
            f.write(adv)
