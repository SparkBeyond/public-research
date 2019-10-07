from guacamol.assess_distribution_learning import *
import datetime
import generateWithTrainedModel


class molGen(DistributionMatchingGenerator):

    def generate(self, number_samples):
        # manually set parameters file
        params = 'params/images/params/80.sess'
        generated_data = generateWithTrainedModel.generate_with_pretrained(params, batch_size=number_samples)
        return generated_data


if __name__ == '__main__':
    # TODO: First remove frechet from guacamol evaluation or load chemnet model after model.generate()
    generator = molGen()
    print('Start Evaluation: ' + str(datetime.datetime.now().time()))
    train_file ='molecules/new_data/guacamol_v1_test.smiles'
    assess_distribution_learning(model=generator, chembl_training_file=train_file,
                                 json_output_file='params/res.json')
    print('End Evaluation: ' + str(datetime.datetime.now().time()))
