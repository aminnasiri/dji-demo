package com.thinksky.classification;

import ai.djl.Model;
import ai.djl.basicdataset.tabular.CsvDataset;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.enterprise.context.ApplicationScoped;
import java.nio.file.Paths;

@ApplicationScoped
public class TrainModel {

    private static final Logger logger = LoggerFactory.getLogger(TrainModel.class);

    private final MakeModel makeModel;
    private final MyPrepareData prepareData;

    public TrainModel(MakeModel makeModel, MyPrepareData prepareData) {
        this.makeModel = makeModel;
        this.prepareData = prepareData;
    }

    public void trainModel() {
//        System.setProperty("offline", "true");
        System.setProperty("ai.djl.default_engine", "PyTorch");
//        System.setProperty("requires_grad", "True");
//        System.setProperty("retain_graph", "False");
        logger.info("You are using: {} Engine", Engine.getInstance().getEngineName());
        try {
            //Creating Training and Testing dataset
            // Prepare the vocabulary
            var ndListNDListZooModel = makeModel.loadWordEmbedding();
            DefaultVocabulary vocabulary = DefaultVocabulary.builder()
                    .addFromTextFile(ndListNDListZooModel.getArtifact("vocab.txt"))
                    .optUnknownToken("[UNK]")
                    .build();
            // Prepare dataset
            int maxTokenLength = 64; // cutoff tokens length
            int batchSize = 4;
//            int limit = Integer.MAX_VALUE;
            int limit = 512; // uncomment for quick testing

            BertFullTokenizer tokenizer = new BertFullTokenizer(vocabulary, true);
            CsvDataset amazonReviewDataset = prepareData.getDataset(batchSize, tokenizer, maxTokenLength, limit);
            // split data with 7:3 train:valid ratio
            RandomAccessDataset[] datasets = amazonReviewDataset.randomSplit(7, 3);
            RandomAccessDataset trainingSet = datasets[0];
            RandomAccessDataset validationSet = datasets[1];

            logger.info("TrainSet {}", amazonReviewDataset);
            logger.info("ValidationSet {}", validationSet);


            //Setup Trainer and training config
            SaveModelTrainingListener listener = new SaveModelTrainingListener("./build/model/qat");
            listener.setSaveModelCallback(
                    trainer -> {
                        TrainingResult result = trainer.getTrainingResult();
                        Model model = trainer.getModel();
                        // track for accuracy and loss
                        float accuracy = result.getValidateEvaluation("Accuracy");
                        model.setProperty("Accuracy", String.format("%.5f", accuracy));
                        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                    });

            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss()) // loss type
                    .addEvaluator(new Accuracy())
                    .optDevices(Engine.getInstance().getDevices(1)) // train using single GPU
                    .addTrainingListeners(TrainingListener.Defaults.logging("./build/model/qat"))
                    .addTrainingListeners(listener);

            //Start training
            int epoch = 2;

            var model = makeModel.makeModel();
            Trainer trainer = model.newTrainer(config);
            trainer.setMetrics(new Metrics());
            Shape encoderInputShape = new Shape(batchSize, maxTokenLength);
            // initialize trainer with proper input shape
            trainer.initialize(encoderInputShape);
            EasyTrain.fit(trainer, epoch, trainingSet, validationSet);

            ndListNDListZooModel.close();

            logger.info("Training Result {}", trainer.getTrainingResult());

            model.setProperty("Epoch", String.valueOf(epoch));
            model.save(Paths.get("./build/model/qat"), "RestaurantQuestionsTag-" + System.currentTimeMillis());
            logger.info("Model {}", model);

//            //Verify the model
//            String review = "I would like to come there, are you open?";
//            Predictor<String, Classifications> predictor = model.newPredictor(new MyTranslator(tokenizer));
//
//            var predict = predictor.predict(review);
//
//            logger.info("Predict JSON {}", predict.toJson());
//            logger.info("Predict Object {}", predict);
//
//            logger.info("Predict Best{}", predict.best());

        } catch (Exception ex) {
            logger.error("Error", ex);
        }

    }
}
