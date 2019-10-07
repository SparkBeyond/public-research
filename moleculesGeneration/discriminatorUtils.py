import numpy as np
import os
import datetime
import pandas as pd
import utilsSG
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
import molecule
import semanticGanArchitecture
import tensorflow as tf
import multiprocessing as mp


def train_discriminator(real, fake, column_names, dp_password, problemName, projectName, as_sequence_ts, size_cap, engine, mappings, number_of_bins, images_dir):
    values = np.concatenate((real, fake))
    isReal = np.concatenate((np.ones(real.shape[0]), np.zeros(fake.shape[0]))).flatten()
    d = {'ts': [list(i) for i in values]} if as_sequence_ts else {column_names[i]: values[:, i] for i in range(len(column_names))}
    d['isReal'] = isReal
    df = pd.DataFrame.from_dict(d)
    df = df.sample(frac=1)
    if df.shape[0] > size_cap:
        df = df.sample(n=size_cap)
    if as_sequence_ts and engine != 'seqgan': df.ts = df.ts.apply(lambda x: [i for i in x if i >= 0 or i < 0])
    df = add_columns_if_needed(df, problemName, engine, as_sequence_ts, mappings)
    if engine == 'scikit':
        working_df = df.fillna(0)
        data = working_df.loc[:, working_df.columns != 'isReal']
        y = np.ravel(working_df.loc[:, working_df.columns == 'isReal'])
        learning_session = GradientBoostingClassifier()
        learning_session.fit(data, y)
        evaluation = cross_validate(learning_session, data, y, cv=3, scoring='roc_auc')['test_score'].min()
    elif engine == 'seqgan':
        working_df = df.fillna(mappings['char2num']['EOS'])
        data = working_df.loc[:, working_df.columns != 'isReal']
        y = np.ravel(working_df.loc[:, working_df.columns == 'isReal'])
        x = np.array([np.array(data.ts.values[i]) for i in range(len(data.ts.values))])
        x[np.isnan(x)] = mappings['char2num']['EOS']
        learning_session, evaluation = train_nn_discriminator(x, y, len(column_names), 200, number_of_bins, images_dir)
    else:
        raise Exception('Illegal engine ' + engine)
    print('new discriminator, auc = ', str(evaluation))
    return learning_session, evaluation


def pred_dist(lst):
    x, learning_session = lst
    return learning_session.predict_proba(np.array([x]))


def predict_with_discriminator(learning_res, x, column_names, as_sequence_ts, problem_name, engine, boundaries):
    df = pd.DataFrame(index=range(x.shape[0]))
    if as_sequence_ts:
        df['ts'] = [list([j for j in i if not np.isnan(j)]) for i in x]
    else:
        for i in range(len(column_names)):
            df[column_names[i]] = x[:, i]
    df_to_predict = add_columns_if_needed(df, problem_name, engine, as_sequence_ts, boundaries)
    if engine == 'SB':
        pred = learning_res.predict(df_to_predict).data
    elif engine == 'scikit':
        data = df_to_predict.fillna(0).loc[:, df_to_predict.columns != 'isReal'].values
        #pred = learning_res.predict_proba(data)
        pool = mp.Pool(mp.cpu_count())
        pred = pool.map(pred_dist, [[i, learning_res] for i in data])
        pool.close()
        pred = pd.DataFrame({'probability_10': np.squeeze(np.array(pred))[:, 1]})  #pred[:, 1]})
    elif engine == 'seqgan':
        data = df_to_predict.loc[:, df_to_predict.columns != 'isReal']
        data = np.array([np.array(i) for i in np.squeeze(data.values)])
        data_pred = learning_res.forward(data)
        pred = learning_res.session.run(data_pred)
        pred = pd.DataFrame({'probability_10': pred[:, 1]})
    return pred


def get_discriminator_rewards(learning_res, fake_samples_discrete, fake_samples, boundaries, number_of_bins, column_names, as_sequence_ts, iter_dynamic_step,
                              priors=None, samples_per_reward=1, rollout=True, problem_name='', engine='illegal', discrete_mode=False, with_partial_rewards=True):
    if with_partial_rewards:
        relevant_data_to_score = []
        dimensions = fake_samples.shape[1]

        keep_inidces = np.append(np.random.choice(range(7, dimensions-1), int(len(range(7, dimensions-1))*.8), replace=False), dimensions-1)
        drop_indices = np.delete(np.arange(dimensions), keep_inidces)

        for dim in range(dimensions):

            if dim in drop_indices: continue

            relevant_data = fake_samples[:, 0:dim + 1]
            if dim != fake_samples.shape[1]:
                if rollout:
                    relevant_data_to_score.append(
                        utilsSG.complete_batch_configuration(boundaries, number_of_bins, relevant_data, dim, priors,
                                                             samples_per_reward))
                else:
                    relevant_data = np.copy(fake_samples)
                    relevant_data[:, dim + 1:] = np.nan if not discrete_mode else boundaries['char2num']['EOS']
                    relevant_data_to_score.append(relevant_data)

        data_to_predict = np.concatenate([relevant_data_to_score[i] for i in range(len(relevant_data_to_score))])
        sb_pred = predict_with_discriminator(learning_res, data_to_predict, column_names, as_sequence_ts, problem_name, engine, boundaries)
        batch_rewards_flat = sb_pred['probability_10'].values
        batch_rewards = np.zeros(fake_samples.shape)
        batch_rewards_by_keep_dim = np.stack(np.split(batch_rewards_flat, len(keep_inidces))).T
        batch_rewards[:, keep_inidces] = batch_rewards_by_keep_dim
    else:
        sb_pred = predict_with_discriminator(learning_res, fake_samples, column_names, as_sequence_ts, problem_name, engine, boundaries)
        batch_rewards = np.zeros(fake_samples.shape)
        batch_rewards[:, -1] = sb_pred['probability_10'].values

    return batch_rewards


def ts_diff(s):
    a = np.array(s)
    c = a[~np.isnan(a)]
    return [c[i] - c[i - 1] for i in range(1, len(c))]


def add_mol_features(mol, nbits_morgan):
    try:
        # list(molecule.Molecule(mol).get_morgan_fp(nbits=nbits))
        instance = molecule.Molecule(mol)
        return [instance.exactMolWt(), instance.logp(), instance.numAtoms(), instance.numBonds(), len(mol), 1]
    except:
        return [-1 for i in range(4)] + [len(mol), 0]


def add_columns_if_needed(df, pname, engine, as_sequence_ts, mappings):
    if engine == 'scikit' and as_sequence_ts:
        new_df = df.copy()
        if 'isReal' in df.columns:
            new_df['isReal'] = df['isReal']
        if pname == 'circleSeq':
            new_df['l'] = df['ts'].apply(lambda x: len(x))
            new_df['l2'] = df['ts'].apply(lambda x: np.sqrt(np.sum([z*z for z in x])))
        elif 'lines' in pname:
            new_df['std_diff'] = df['ts'].apply(lambda x: np.std(ts_diff(x)))
            new_df['tda'] = df['ts'].apply(lambda x: np.mean(ts_diff(x)))
            new_df['len'] = df['ts'].apply(lambda x: len(x))
        elif pname == 'mol' and engine == 'scikit':
            unique_chars = mappings['char2num'].keys()

            def add_BOW(smiles):
                counts = {i: 0 for i in unique_chars}
                for i in list(smiles): counts[i] += 1
                return counts.values()

            dimensions = max([len(i) for i in df.ts.values])  # todo: pass dimensions
            new_df['smiles'] = df.ts.apply(lambda x: utilsSG.seq_2_smile(x, mappings['num2char']))
            new_df['partial'] = df.ts.apply(lambda x: len(x) < dimensions)
            new_df.reset_index(drop=True, inplace=True)
            smiles_lst = new_df.smiles.values
            pool = mp.Pool(mp.cpu_count())
            results = pool.map(add_mol_features_dist, smiles_lst)
            pool.close()
            new_df['features'] = results
            new_df[['weight', 'logP', 'BertzCT', 'TPSA', 'numHAcceptors', 'numHDonors', 'NumRotatableBonds', 'NumAliphaticRings',
                    'NumAromaticRings', 'smileLen', 'valid']] = pd.DataFrame(new_df.features.tolist())

            new_df['bow'] = new_df.smiles.apply(add_BOW)
            new_df[['bow_char_' + i for i in unique_chars]] = pd.DataFrame(new_df.bow.tolist())
            new_df.drop(['smiles', 'ts', 'features', 'bow'], axis=1, inplace=True)
    else:
        new_df = df
    return new_df


def train_nn_discriminator(data, y, dimensions, state_size, number_of_bins, images_dir):

    X = tf.placeholder(tf.int32, shape=[None, dimensions])
    Y = tf.placeholder(tf.float32, shape=[None, 2])
    NN = semanticGanArchitecture.RnnDiscriminator(state_size, dimensions, number_of_bins)

    predictions = NN.forward(X)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=predictions)
    total_loss = tf.reduce_mean(loss)

    learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)

    if os.path.isfile(images_dir + '/params/discriminator_params.sess.meta'):
        saver.restore(sess, os.path.normpath(images_dir + '/params/discriminator_params.sess'))

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 42)
    y_test_one_hot = np.array(OneHotEncoder(categories=[range(2)]).fit_transform(np.expand_dims(y_test, -1)).todense())
    validation_loss = []
    best_validation = np.inf
    for i in range(100):
        x_batch, y_batch = utilsSG.get_labeled_batch(X_train, y_train, 128)
        label_batch = np.array(OneHotEncoder(categories=[range(2)]).fit_transform(np.expand_dims(y_batch, -1)).todense())
        _, iter_pred, iter_losses, iter_mean_loss = sess.run([optimizer, predictions, loss, total_loss], feed_dict={X: x_batch, Y: label_batch})
        if i % 10 == 0:
            validation_loss.append(sess.run(total_loss, feed_dict={X: X_test, Y: y_test_one_hot}))
            if validation_loss[-1] < best_validation:
                best_validation = validation_loss[-1]
                saver.save(sess, os.path.normpath(images_dir + '/params/discriminator_params.sess'))
            if len(validation_loss) > 50 and np.all(validation_loss[-2:] > best_validation):
                break

    saver.restore(sess, os.path.normpath(images_dir + '/params/discriminator_params.sess'))

    y_test_pred = sess.run(predictions, feed_dict={X: X_test, Y: y_test_one_hot})[:, 1]

    auc = roc_auc_score(y_test, y_test_pred)

    NN.session = sess

    return NN, auc


def add_mol_features_dist(mol, num_of_features=9):
    try:
        instance = molecule.Molecule(mol)
        return [instance.exactMolWt(), instance.logp(), instance.BertzCT(), instance.TPSA(), instance.numHAcceptors(),
                instance.numHDonors(), instance.NumRotatableBonds(), instance.NumAliphaticRings(),
                instance.NumAromaticRings(),
                len(mol), 1]

    except:
        return [-1 for i in range(num_of_features)] + [len(mol), 0]


def generator_auc(real_data, generated_data, mappings):
    print(str(datetime.datetime.now().time()))
    new_df = pd.concat([real_data, generated_data])
    new_df['labels'] = np.concatenate([np.zeros(real_data.shape[0]), np.ones(generated_data.shape[0])])
    new_df.reset_index(inplace=True, drop=True)
    labels = new_df.labels
    pool = mp.Pool(mp.cpu_count())
    features = pool.map(add_mol_features_dist, new_df.smiles)
    pool.close()

    new_df[['weight', 'logP', 'BertzCT', 'TPSA', 'numHAcceptors', 'numHDonors', 'NumRotatableBonds', 'NumAliphaticRings',
         'NumAromaticRings', 'smileLen', 'valid']] = pd.DataFrame(np.array(features)) #pd.DataFrame(features.tolist())

    unique_chars = mappings['char2num'].keys()

    def add_BOW(smiles):
        counts = {i: 0 for i in unique_chars}
        for i in list(smiles): counts[i] += 1
        return counts.values()

    new_df['bow'] = new_df.smiles.apply(add_BOW)
    new_df[['bow_char_' + i for i in unique_chars]] = pd.DataFrame(new_df.bow.tolist())
    new_df.drop(['smiles', 'bow', 'labels'], axis=1, inplace=True)

    learning_session = GradientBoostingClassifier()
    learning_session.fit(new_df, labels)
    evaluation = cross_validate(learning_session, new_df, labels, cv=3, scoring='roc_auc')['test_score'].min()
    return evaluation

