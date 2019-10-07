import os
import numpy as np
import molecule
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
import datetime
import pickle


def get_data(file):
    data = pd.read_csv(file, sep='\t')
    return data.values, data.shape[1], data, data.columns.values


def get_batch(data, size):
    return data[np.random.choice(data.shape[0], size=size)]


def get_labeled_batch(data, labels, size):
    indices = np.random.choice(data.shape[0], size=size)
    return data[indices], labels[indices]


def boundaries_and_discretezation(data, bins, dimensions):
    cutoffs = []
    for i in range(dimensions):
        empirical_min, empricial_max = data[:, i].min(), data[:, i].max()
        step = float(empricial_max - empirical_min) / bins[i]
        cutoffs.append([empirical_min + i * step for i in range(bins[i] + 1)])
    discrete = [np.digitize(data[:, i], cutoffs[i][1:-1], right=True) for i in range(len(cutoffs))]
    return np.array(discrete), np.array(cutoffs)


def discretize(data, cutoffs):
    discrete = [np.digitize(data[:, i], cutoffs[i][1:-1], right=True) for i in range(min(len(cutoffs), data.shape[1]))]
    return np.array(discrete)


def reverse_discretiztion(discrete_data, cutoffs, randomize=True):
    continous_data = []
    for i in range(discrete_data.shape[1]):
        point = []
        for dim in range(discrete_data.shape[0]):
            left = cutoffs[dim][discrete_data[dim, i]]
            right = cutoffs[dim][discrete_data[dim, i] + 1]
            if randomize:
                point.append(np.random.uniform(low=left, high=right))
            else:
                point.append(np.mean([left, right]))
        continous_data.append(point)
    return np.array(continous_data)


def reverse_discretiztion_tf(discrete_data, cutoffs, dim, randomize=False):
    left = tf.gather(cutoffs[dim-1], discrete_data)
    right = tf.gather(cutoffs[dim-1], discrete_data + 1)
    mean_bin = tf.reduce_mean(tf.concat([left, right], axis=1), axis=1, keepdims=True)
    return tf.cast(mean_bin, tf.float32)


def complete_batch_configuration(boundaries, bins, partial_assignment, assignment_dim, priors=None):
    complete_assignment = np.zeros((partial_assignment.shape[0], len(bins)))
    complete_assignment[:, :assignment_dim + 1] = partial_assignment
    for dim in range(assignment_dim + 1, len(bins)):
        sample_bin = np.random.choice(a=range(bins[dim]), size=partial_assignment.shape[0])
        sample_dim_instances = [np.random.uniform(low=boundaries[dim][sample_bin[i]],
                                                  high=boundaries[dim][sample_bin[i] + 1]) for i in range(partial_assignment.shape[0])]
        complete_assignment[:, dim] = sample_dim_instances
    return complete_assignment


def expand_data_with_partial_path(df):
    new_df = df
    for i in range(1, df.shape[1]):
        partial = np.zeros(shape=df.shape)
        partial[:, :i] = df[:, :i]
        partial[:, i:] = np.NaN
        new_df = np.concatenate([new_df, partial[np.random.choice(partial.shape[0], int(partial.shape[0]/30))]])
    return new_df


# discretize the data and then project it back to the original space
# (to cancel out discretization effects from actual data)
def actual_data_reprojected_if_needed(data, cutoffs, actual_points_discretized):
    if actual_points_discretized:
        bin_numbers = discretize(data, cutoffs)
        return reverse_discretiztion(bin_numbers, cutoffs)
    else:
        return data


def validate_stop_criteria(auc):
    if auc < 0.53:
        print('AUC = ', auc, ', Generated data and real data are inseparable')
        return True
    return False


def visualize_extrapolated_ts(images_dir, extrapulated_samples, number_of_sequences=9, fn='extrapolated_samples.png'):
    s = pd.DataFrame(extrapulated_samples[0: min([number_of_sequences, extrapulated_samples.shape[0]]), :])
    s.T.plot(subplots=True, legend=False)
    plt.savefig(images_dir + '/' + fn)
    plt.close()


def print_status(train_dir, images_dir, data, it, fake_samples, real_data_nll, fake_data_nll, likelihoods, mean_rewards, aucs, multi_dim=True, sequences=False, problem_name='', boundaries={}, working_actual_smiles=[]):
    images_dir = images_dir.split('.tsv')[0]
    fake_samples = fake_samples.squeeze()

    plt.plot(range(len(fake_data_nll)), fake_data_nll)
    plt.savefig(images_dir + "/fake_data_nll.png")
    plt.close()

    plt.plot(range(len(real_data_nll)), real_data_nll)
    plt.savefig(images_dir + "/real_data_nll.png")
    plt.close()

    plt.plot(range(len(likelihoods)), likelihoods)
    plt.savefig(images_dir + "/likelihoods.png")
    plt.close()

    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.savefig(images_dir + "/mean_rewards.png")
    plt.close()

    plt.plot(range(len(aucs)), aucs)
    plt.savefig(images_dir + "/auc.png")
    plt.close()

    if it == 0:
        pd.DataFrame(data).to_csv(images_dir + '/actual_samples_.tsv', sep='\t', index=False)
        pd.DataFrame(working_actual_smiles.smiles).to_csv(images_dir + '/actual_samples_smiles.tsv', sep='\t', index=False)

    if fake_samples.size > 0:
        samples_df = pd.DataFrame(fake_samples)
        samples_df.to_csv(images_dir + '/generated_samples_' + str(it) + ".tsv", sep='\t', index=False)
        samples_df.to_csv(train_dir + '/_generated_data.tsv', sep='\t')
    else:
        print("Warning: no fake samples for iteration " + str(it))

    if problem_name == 'circleSeq':
        plt.scatter(x=data[:, 0], y=data[:, 1], c='r', marker='x', label='Real')
        plt.scatter(x=fake_samples[:, 0], y=fake_samples[:, 1], c='g', marker='s', label='Generated')
        plt.legend(loc='upper left')
        plt.savefig(images_dir + "/generated_" + str(it) + ".png")
        plt.close()
    elif sequences:
        number_of_lines = 7
        s = pd.DataFrame(fake_samples[0:number_of_lines, :])
        s.T.plot(subplots=True, legend=False)
        plt.savefig(images_dir + "/generated_" + str(it) + ".png")
        plt.close()
    elif not multi_dim:
        sns.distplot(data, label='Real', color='r', norm_hist=True)
        sns.distplot(fake_samples, label='Generated', color='g', norm_hist=True)
        plt.savefig(images_dir + "/generated_" + str(it) + ".png")
        plt.close()
    elif data.shape[1]==2:
        plt.scatter(x=data[:, 0], y=data[:, 1], c='r', marker='x', label='Real')
        plt.scatter(x=fake_samples[:, 0], y=fake_samples[:, 1], c='g', marker='s', label='Generated')
        plt.legend(loc='upper left')
        plt.savefig(images_dir + "/generated_" + str(it) + ".png")
        plt.close()
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], c='r', marker='x', label='Real')
        ax.scatter(xs=fake_samples[:, 0], ys=fake_samples[:, 1], zs=fake_samples[:, 2], c='g', marker='s', label='Generated')
        ax.legend(loc='upper left')
        plt.savefig(images_dir + "/generated_" + str(it) + ".png")
        plt.close()


def get_molecule_data_set(load_from_disc_path, training_set_size, load_all=True):

    if load_all: return load_data_and_mappings_from_disc(training_samples=training_set_size)

    data = pd.read_csv(load_from_disc_path, header=None, names=['smiles'])

    short_mol = data[data.smiles.str.len() <= 55]
    print(datetime.datetime.now())
    data = short_mol.smiles.apply(lambda x: pd.Series(list(x)))
    data = data.fillna("EOS")
    unique_characters = pd.unique(data.values.ravel())
    num2char = {index: v for index, v in enumerate(unique_characters)}
    char2num = {v: index for index, v in enumerate(unique_characters)}
    mappings = {"num2char": num2char, "char2num": char2num}
    assert (np.all([char2num[num2char[i]] == i for i in num2char.keys()]))

    data = data.applymap(lambda x: char2num[x])

    with open(os.path.join(os.path.join(os.getcwd(), 'molecules/new_data/data_and_mappings.pickle')), 'wb') as handle:
        pickle.dump([data, mappings], handle)

    print(datetime.datetime.now())
    return data.values, data.shape[1], data, data.columns.values, mappings


def load_data_and_mappings_from_disc(training_samples, loc=os.path.join(os.path.join(os.getcwd(), 'molecules/new_data/data_and_mappings.pickle'))):
    with open(loc, 'rb') as handle:
        data, mappings = pickle.load(handle)
        data = data.sample(training_samples + 5000) # 5k for validation
    return data.values, data.shape[1], data, data.columns.values, mappings


def discrete_batch_to_str_batch(batch, num2char):
    smiles = []
    for sample in batch:
        chars = map(lambda x: num2char[x], sample)
        smiles.append(''.join([i for i in chars if i != 'EOS']))
    return smiles


def eval_molecule_batch_validity_rate(batch):
    import molecule
    valid_molecules = 0
    for mol in batch:
        try:
            if mol == '': continue
            molecule.Molecule(mol)
            valid_molecules += 1
        except: continue
    valid_molecules_rate = float(valid_molecules) / len(batch)
    return valid_molecules_rate


def molecule_validity(mol):
    try:
        if mol == '': return 0
        molecule.Molecule(mol)
        return 1
    except:
        return 0


def seq_2_smile(seq, num2char):
    chars = map(lambda x: num2char[x], seq)
    return ''.join([i for i in chars if i != 'EOS'])


def print_mol_status(kl_over_time, validity_rates, nll_over_time, smiles_batch, images_dir, iteration, props_over_time={}):
    print("Valid molecules rate {}".format(validity_rates[-1]))
    plt.plot(range(len(validity_rates)), validity_rates)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.savefig(images_dir + "/validity.png")
    plt.close()

    plt.plot(range(len(props_over_time['logP'])), props_over_time['logP'])
    plt.savefig(images_dir + "/logP.png")
    plt.close()

    plt.plot(range(len(props_over_time['weight'])), props_over_time['weight'])
    plt.savefig(images_dir + "/weight.png")
    plt.close()

    pd.DataFrame(smiles_batch.smiles).to_csv(images_dir + '/generated_smiles_' + str(iteration) + '.tsv', sep='\t', index=False)

    plt.plot(range(len(kl_over_time['weight'])), kl_over_time['weight'])
    plt.savefig(images_dir + "/kl_weight.png")
    plt.close()

    plt.plot(range(len(kl_over_time['logP'])), kl_over_time['logP'])
    plt.savefig(images_dir + "/kl_logP.png")
    plt.close()

    pd.DataFrame(np.array([nll_over_time, validity_rates, kl_over_time['logP'], kl_over_time['weight']]).T,
                 columns=["NLL", 'VALID', 'LOGP KL', 'WEIGHT KL']).to_csv(images_dir + "/numeric_evaluation.csv")

