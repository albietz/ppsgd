import argparse
import pickle

# download from https://github.com/google-research/federated/blob/master/utils/datasets/stackoverflow_tag_prediction.py
import stackoverflow_tag_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser('S-O data')
    parser.add_argument('--word_vocab_size', type=int, default=5000)
    parser.add_argument('--tag_vocab_size', type=int, default=80)
    parser.add_argument('--num_clients', type=int, default=500)
    args = parser.parse_args()

    tr, te = stackoverflow_tag_prediction.get_federated_datasets(word_vocab_size=args.word_vocab_size, tag_vocab_size=args.tag_vocab_size)

    train_datasets = []
    test_datasets = []
    for cid in te.client_ids[:args.num_clients]:
        Xs, Ys = [], []
        for elem in tr.create_tf_dataset_for_client(cid):
            Xs.append(elem[0].numpy())
            Ys.append(elem[1].numpy())

        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        train_datasets.append((X, Y))
    pickle.dump(train_datasets, open('stackoverflow_train.pkl', 'wb'))

    for cid in te.client_ids[:args.num_clients]:
        Xs, Ys = [], []
        for elem in te.create_tf_dataset_for_client(cid):
            Xs.append(elem[0].numpy())
            Ys.append(elem[1].numpy())

        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        test_datasets.append((X, Y))
    pickle.dump(test_datasets, open('stackoverflow_test.pkl', 'wb'))
