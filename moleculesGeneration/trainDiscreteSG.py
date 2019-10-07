import utilsSG
import semanticGanArchitecture
import discriminatorUtils
from sklearn.model_selection import train_test_split
import argparse
import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import datetime
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def train(train_dir,
          images_dir,
          batch_size=10000,
          print_status_per=10,
          initial_iter_per_discrimnator=20,
          number_of_bins_per_dim=23,
          max_iterations=100000,
          slices_per_batch=10,
          actual_points_discretized=True,
          use_partial_path=True,
          single_class_mode=1,
          problem_name='',
          project_name='',
          dp_password='somePass',
          as_sequence_ts=True,
          sequences_to_complete_fn=None,
          sb_size=100000,
          engine='SB',
          dynamic_seq_length_mode=True,
          supervised_pre_training_steps=0,
          discrete_data_mode=True,
          load_supervised_param=False,
          training_set_size=0):

    main_graph = tf.Graph()
    with main_graph.as_default():
        data, dimensions, train_DF, column_names, boundaries = utilsSG.get_molecule_data_set(os.path.join(os.path.join(os.getcwd(), 'molecules/new_data'), 'guacamol_v1_train.smiles'), training_set_size)
        number_of_bins = [len(boundaries['char2num']) for i in range(data.shape[1])] if discrete_data_mode else [number_of_bins_per_dim for i in range(data.shape[1])]
        train_set, validation_set = train_test_split(data, test_size=5000, random_state=1)
        samples_per_slice = int(np.ceil(float(batch_size) / slices_per_batch))
        X = tf.placeholder(tf.int32, shape=[None, dimensions], name='X')
        rewards = tf.placeholder(tf.float32, shape=[None, dimensions], name='reward')
        dynamic_extrapolate_steps = tf.placeholder(tf.int32, shape=(None,), name="dynamic_steps")
        NN = semanticGanArchitecture.SemanticGanNet(number_of_bins, dimensions, batch_size, numeric_input_mode=False)

        if dynamic_seq_length_mode:
            complete_seq_logits_pre, complete_seq_states_pre, complete_seq_samples_pre = NN.generate()
            complete_seq_logits_pre = tf.stack(complete_seq_logits_pre, axis=1)  # batch x dimensions x bins
            complete_seq_gen_samples = tf.cast(complete_seq_samples_pre, dtype=tf.int64)
            logits_pre, states_pre, samples_pre = NN.dynamic_extrapolate_from_input(X, dynamic_extrapolate_steps)

        else:
            logits_pre, states_pre, samples_pre = NN.dynamic_extrapolate_from_input(X, tf.zeros(tf.shape(X)[0], dtype=np.int32))

        logits_pre = tf.stack(logits_pre, axis=1)  # batch x dimensions x bins
        gen_samples = tf.cast(samples_pre, dtype=tf.int64)

        logits_post, states_post, samples_post = NN.generate_with_supervision(X)
        logits = tf.stack(logits_post, axis=1)

        current_batch_size = tf.cast(tf.shape(X)[0], dtype=tf.float32)
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.99, staircase=True)
        G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)

        loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(X, [-1])), number_of_bins[0], 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(logits, [-1, number_of_bins[0]]), 1e-20, 1.0)
                ), 1) * tf.reshape(rewards, [-1])
        )

        nll = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(X, [-1])), number_of_bins[0], 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(logits, [-1, number_of_bins[0]]), 1e-20, 1.0)
                ), 1)
        ) / (current_batch_size*dimensions)

        likelihood = tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(X, [-1])), number_of_bins[0], 1.0, 0.0) *
                    tf.clip_by_value(tf.reshape(logits, [-1, number_of_bins[0]]), 1e-20, 1.0), 1)
        ) / (current_batch_size*dimensions)

        grad_and_vars = G_solver.compute_gradients(loss)
        train_op = G_solver.apply_gradients(grad_and_vars, global_step=global_step)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)

        # supervised pre-training
        supervised_grad_and_vars = G_solver.compute_gradients(nll)
        supervised_train_op = G_solver.apply_gradients(supervised_grad_and_vars, global_step=global_step)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        best_real_data_nll = np.inf
        switch_discriminator = False
        real_data_nll, fake_data_nll, likelihoods, mean_rewards, well_separated_examples, fake_samples, aucs = [], [], [], [], [], [], []
        working_actual_data = train_set[np.random.choice(train_set.shape[0], size=int(batch_size/2))]
        working_actual_smiles = pd.DataFrame(utilsSG.discrete_batch_to_str_batch(working_actual_data, boundaries['num2char']), columns=['smiles'])
        working_actual_smiles.reset_index(inplace=True, drop=True)
        working_actual_smiles['features'] = working_actual_smiles.smiles.apply(lambda x: discriminatorUtils.add_mol_features(x, nbits_morgan=32))
        working_actual_smiles[['weight', 'logP', 'nAtoms', 'nBons', 'smileLen', 'valid']] = pd.DataFrame(working_actual_smiles.features.tolist())

        # Supervised pre training
        supervised_batch_size = 256
        validity_rates, nll_over_time = [], []
        best_nll = np.inf
        patience = 0
        for i in range(supervised_pre_training_steps):
            x_batch = utilsSG.get_batch(train_set, supervised_batch_size)

            _, iter_grad, iter_output, iter_samples, iter_nll = sess.run(
                [supervised_train_op, supervised_grad_and_vars, logits, samples_post, nll],
                feed_dict={X: x_batch})

            if i % 100 == 0:
                validation_nll = sess.run(nll, feed_dict={X:validation_set})
                print("Supervised iteration {} train nll {} validation nll {} ".format(i, iter_nll, validation_nll) + str(datetime.datetime.now().time()))
                iter_samples = sess.run(gen_samples, feed_dict={X: x_batch, dynamic_extrapolate_steps: np.zeros(supervised_batch_size)})
                smiles_batch = utilsSG.discrete_batch_to_str_batch(iter_samples.T, boundaries['num2char'])
                valid_mol_rate = utilsSG.eval_molecule_batch_validity_rate(smiles_batch)
                print("Validity rates: {}".format(valid_mol_rate))
                if i % 400 == 0:
                    if validation_nll < best_nll:
                        best_nll = validation_nll
                        best_params = os.path.normpath(images_dir + '/params/supervised_{}.sess'.format(i))
                        saver.save(sess, best_params)
                        patience = 0
                    else:
                        patience += 1
                    if patience > 5: break

        # post supervised pre-training evaluation
        if supervised_pre_training_steps > 0:
            saver.restore(sess, best_params)

        # Semantic adversarial training
        for it in range(max_iterations):
            if it % initial_iter_per_discrimnator == 0 or switch_discriminator:
                try:
                    working_actual_data = train_set[np.random.choice(train_set.shape[0], size=int(batch_size / 2))]
                    fake_discrete_training_data = sess.run(complete_seq_gen_samples).T
                    fake_discrete_training_data_batch = fake_discrete_training_data[np.random.choice(fake_discrete_training_data.shape[0], size=int(batch_size / 2))]
                    if use_partial_path:
                        fake_discrete_training_data_batch = utilsSG.expand_data_with_partial_path(fake_discrete_training_data_batch)
                        working_actual_data = utilsSG.expand_data_with_partial_path(working_actual_data)
                    if it > 0:
                        old = previous_fake_training_data[np.random.choice(previous_fake_training_data.shape[0], size=int(batch_size * .35))]
                        new = fake_discrete_training_data_batch[np.random.choice(fake_discrete_training_data_batch.shape[0], size=int(batch_size * .15))]
                        working_actual_data = working_actual_data[np.random.choice(working_actual_data.shape[0], size=int(batch_size/2))]
                        fake_discrete_training_data_batch = np.concatenate([old, new])
                    learning_res, evaluation = discriminatorUtils.train_discriminator(working_actual_data, fake_discrete_training_data_batch, column_names, dp_password, problem_name, project_name, as_sequence_ts, sb_size, engine, boundaries, number_of_bins, images_dir)
                    previous_fake_training_data = np.copy(fake_discrete_training_data_batch)
                    if utilsSG.validate_stop_criteria(evaluation):
                        break
                except Exception as e:
                    print(e)
                    return column_names, fake_samples, well_separated_examples
                switch_discriminator = False
                saver.save(sess, images_dir + '/params/' + str(it) + '.sess')
                aucs.append(evaluation)

            if it % print_status_per == 0:
                fake_discrete_samples_print = sess.run(complete_seq_gen_samples).T
                smiles_batch = pd.DataFrame(utilsSG.discrete_batch_to_str_batch(fake_discrete_samples_print, boundaries['num2char']), columns=['smiles'])
                validity_rates.append(utilsSG.eval_molecule_batch_validity_rate(smiles_batch.smiles.values))

                real_data_logits, real_data_likelihood, test_nll = sess.run([logits, likelihood, nll], feed_dict={X: validation_set})
                real_data_nll.append(test_nll)
                utilsSG.print_status(train_dir, images_dir, working_actual_data, it, fake_discrete_samples_print[np.random.choice(fake_discrete_samples_print.shape[0], size=1000), :], real_data_nll, fake_data_nll, likelihoods, mean_rewards, aucs, multi_dim=dimensions > 1, sequences=as_sequence_ts, problem_name=problem_name, boundaries=boundaries, working_actual_smiles=working_actual_smiles)
                print("Valid molecules rate {}".format(validity_rates[-1]))
                if real_data_nll[-1] < best_real_data_nll:
                    best_real_data_nll = real_data_nll[-1]
                    saver.save(sess, images_dir + '/params/' + str(it) + '.sess')

            train_set_for_reward_optimization = train_set[np.random.choice(train_set.shape[0], size=batch_size)]
            dynamic_step = [np.where(np.random.multinomial(1, pvals=[0.8] + [0.2/(dimensions-1) for i in range(dimensions-1)]))[0][0] for i in range(batch_size)]
            iter_logit, fake_discrete_samples, iter_learning_rate = sess.run([logits_pre, gen_samples, learning_rate], feed_dict={X: train_set_for_reward_optimization, dynamic_extrapolate_steps: dynamic_step})
            batch_reward = discriminatorUtils.get_discriminator_rewards(learning_res, fake_discrete_samples, fake_discrete_samples.T, boundaries, number_of_bins, column_names, rollout=False, as_sequence_ts=as_sequence_ts, iter_dynamic_step=dynamic_step, problem_name=problem_name, engine=engine, discrete_mode=discrete_data_mode, with_partial_rewards=use_partial_path)

            purly_generated_batch_reward = batch_reward[np.array(dynamic_step) == 0]
            mean_reward = purly_generated_batch_reward[:, -1].mean() #purly_generated_batch_reward[purly_generated_batch_reward != 0].mean() #assume no zero probabilities
            mean_rewards.append(mean_reward)
            if mean_reward > .5:
                print("switching discriminator due to high reward. iter = ", it, " reward = ", mean_reward)
                switch_discriminator = True
                continue

            for i in range(slices_per_batch):
                batch_reward_slice = batch_reward[i*samples_per_slice: (i+1)*samples_per_slice, :]
                fake_discrete_slice = fake_discrete_samples[:, i*samples_per_slice: (i+1)*samples_per_slice]
                _, iter_grad, iter_loss, iter_output, iter_state, iter_samples, iter_nll = sess.run([train_op, grad_and_vars, loss, logits, states_post, samples_post, nll], feed_dict={rewards: batch_reward_slice, X: fake_discrete_slice.T})

            fake_data_nll.append(iter_nll)

            print("iter = ", it, " mean reward = ", mean_reward, " real data nll = ", test_nll, "learning rate = ", iter_learning_rate, 'time = ', str(datetime.datetime.now().time()))

        saver.save(sess, images_dir + '/params/' + str(it) + '_Final.sess')

        return train_DF, fake_samples, well_separated_examples, [], []


def post_training(train_df, generated_samples, well_separated_samples, actual_data_discrete, single_class_mode, singleClassdir, outputFN, extrapolated_data, extrapolation_fn, imageDir):

    if generated_samples.size > 0:
        print("saving generated samples to " + outputFN)
        generated_df = pd.DataFrame(generated_samples, columns=train_df.columns.values)
        generated_df.to_csv(outputFN, sep='\t', index=False)
    else:
        print("Warning: empty generated samples after train")

    if extrapolated_data.size > 0:
        ofn = imageDir + '/extrapolated.tsv'
        print("saving extrapolated samples to " + ofn)
        extrapolated_df = pd.DataFrame(extrapolated_data, columns=train_df.columns.values)
        extrapolated_df.to_csv(ofn, sep='\t', index=False)

    if single_class_mode > 0:
        def singleClassDataset(realSamplesDF, fakeSamples):
            actual = realSamplesDF.copy()
            actual["isReal"] = True
            fake = pd.DataFrame(fakeSamples, columns=realSamplesDF.columns.values)
            fake["isReal"] = False
            dataset = pd.concat([actual, fake])
            return dataset #we do not shuffle so the first fake examples are from the first iterations
        print("generating single class datasets")
        singleClassDFActual = singleClassDataset(train_df, well_separated_samples)
        singleClassDFActual.to_csv(singleClassdir + 'singleClassActual.tsv', sep='\t', index=False)
        singleClassDFDiscrete = singleClassDataset(pd.DataFrame(actual_data_discrete, columns=train_df.columns.values), actual_data_discrete)
        singleClassDFDiscrete.to_csv(singleClassdir + 'singleClassDiscrete.tsv', sep='\t', index=False)


def run(args):
    train_dir = args.output_path
    imageDir = train_dir + '/images'
    if os.path.exists(imageDir): shutil.rmtree(imageDir)
    os.makedirs(imageDir)
    os.makedirs(imageDir + '/params')
    outputFN = train_dir + '_generated_data.tsv'
    password, max_iterations, number_of_bins_per_dim, single_class_mode, problem_name, sb_train_size = \
        args.password, int(args.max_iterations), int(args.number_of_bins), int(args.singleClassMode), \
        args.problemName, int(args.sb_train_size)
    actual_points_discretized = True
    generate_sequences = args.generate_sequences == "True"
    extrapolation_fn = train_dir + args.sequences_to_complete_fn if args.sequences_to_complete_fn else ''
    batch_core = 4000 if generate_sequences else 10000
    engine = args.engine
    assert (engine in ['SB', 'scikit', 'seqgan']), 'illegal engine ' + engine

    # train the generator
    train_df, generated_samples, well_separated_samples, actual_data_discrete, extrapolated_data = train(
        train_dir=train_dir,
        images_dir=imageDir,
        batch_size=batch_core,
        number_of_bins_per_dim=43,
        max_iterations=1000000,
        actual_points_discretized=True,
        single_class_mode=single_class_mode,
        problem_name=problem_name,
        project_name='separation' + problem_name,
        dp_password=password,
        as_sequence_ts=True,
        sequences_to_complete_fn=extrapolation_fn,
        sb_size=sb_train_size,
        engine=engine,
        dynamic_seq_length_mode=True,
        supervised_pre_training_steps=int(args.mle_iterations),
        load_supervised_param=True,
        training_set_size=int(args.train_set_size)
    )
    # save results and build single class
    post_training(train_df, generated_samples, well_separated_samples, actual_data_discrete, single_class_mode, imageDir, outputFN, extrapolated_data, extrapolation_fn, imageDir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data with semantics')
    parser.add_argument('-o', '--output_path',  required=False, default="params")
    parser.add_argument('-s', '--singleClassMode', required=False, default="0")
    parser.add_argument('-p', '--password', required=False, default="somePass")
    parser.add_argument('-m', '--max_iterations', required=False, default="1000")
    parser.add_argument('-n', '--number_of_bins', required=False, default="43")
    parser.add_argument('-pn', '--problemName', required=False, default="mol")
    parser.add_argument('-ad', '--actualPointsDiscretized', required=False, default="True")
    parser.add_argument('-seq', '--generate_sequences', required=False, default="True")
    parser.add_argument('-sc', '--sequences_to_complete_fn', required=False, default="")
    parser.add_argument('-sbsize', '--sb_train_size', required=False, default="100000")
    parser.add_argument('-e', '--engine', required=False, default="scikit")
    parser.add_argument('-ts', '--train_set_size', required=False, default="10000")
    parser.add_argument('-mle', '--mle_iterations', required=False, default="10000")
    args = parser.parse_args()
    run(args)
