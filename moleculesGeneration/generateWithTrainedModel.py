import tensorflow as tf
import pandas as pd
import utilsSG
import semanticGanArchitecture
import argparse
import os


def generate_with_pretrained(param_file, batch_size, output_dir=None):
    # TODO : load boundaries separately
    real_data, dimensions, train_DF, column_names, boundaries = utilsSG.get_molecule_data_set(
        os.path.join(os.path.join(os.getcwd(), 'molecules/new_data'), 'guacamol_v1_train.smiles'), 1000)
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            number_of_bins = [len(boundaries['char2num']) for i in range(real_data.shape[1])]
            NN = semanticGanArchitecture.SemanticGanNet(number_of_bins=number_of_bins, dimensions=dimensions, batch_size=batch_size, numeric_input_mode=False)
            logits, states, samples = NN.generate()
            gen_samples = tf.cast(samples, dtype=tf.int64)

            #sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
            saver.restore(sess, param_file)

            iter_samples = sess.run(gen_samples, feed_dict={})
            smiles_batch = utilsSG.discrete_batch_to_str_batch(iter_samples.T, boundaries['num2char'])

    if output_dir is not None:
        pd.DataFrame(smiles_batch).to_csv(output_dir)
    print("Done Generating {}".format(batch_size))

    return smiles_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate with trained model')
    parser.add_argument('-p', '--parameter_file')
    parser.add_argument('-b', '--batch_size', required=True)
    parser.add_argument('-o', '--output_dir', required=True)

    args = parser.parse_args()
    generate_with_pretrained(args.parameter_file, int(args.batch_size), args.output_dir)

