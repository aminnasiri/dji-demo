package com.thinksky.classification;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.enterprise.context.ApplicationScoped;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


@ApplicationScoped
public class MakeModel {

    private static final Logger logger = LoggerFactory.getLogger(MakeModel.class);

    private final MyPrepareData prepareData;

    public MakeModel(MyPrepareData prepareData) {
        this.prepareData = prepareData;
    }

    public String getClassificationGroup(String question) throws TranslateException, IOException {

        var ndListNDListZooModel = this.loadWordEmbedding();
        DefaultVocabulary vocabulary = DefaultVocabulary.builder()
                .addFromTextFile(ndListNDListZooModel.getArtifact("vocab.txt"))
                .optUnknownToken("[UNK]")
                .build();

        var myTranslator = new MyTranslator(new BertFullTokenizer(vocabulary, true));

//        var criteria = Criteria.builder()
//                .optApplication(Application.NLP.WORD_EMBEDDING)
//                .setTypes(NDList.class, NDList.class)
////                .optModelPath(Paths.get("build/model/qat/RestaurantQuestionsTag"))
////                .optModelName("RestaurantQuestionsTag")
////                .optTranslator(myTranslator)
//                .build();
//
//        try (var model = ModelZoo.loadModel(criteria)) {
//
//            model.load(Paths.get("build/model/qat/"), "RestaurantQuestionsTag");
//            Predictor<String, Classifications> predictor = model.newPredictor(myTranslator);
//
//            var predict = predictor.predict(question);
//
//            logger.info("Predict JSON {}", predict.toJson());
//            logger.info("Predict Object {}", predict);
//
//            logger.info("Predict Best{}", predict.best());
//
//            return predict.best().getClassName();
//        } catch (MalformedModelException | IOException | ModelNotFoundException e) {
//            throw new RuntimeException(e);
//        }

        try (Model model = Model.newInstance("RestaurantQuestionsTag")) {

            model.setBlock(this.getBlock());

            // assume that you have train and save your model in build/model folder.
            model.load(Paths.get("build/model/qat/"), "RestaurantQuestionsTag");


            try (Predictor<String, Classifications> predictor = model.newPredictor(myTranslator)) {
                var className = predictor.predict(question).best().getClassName();
                ndListNDListZooModel.close();
                return className;
            }
        } catch (MalformedModelException | IOException e) {
            logger.error("Prediction was not go well.", e);
            throw new RuntimeException(e);
        }

    }

    public List<String> listOfModel() {
        try {
            var models = ModelZoo.listModels();
            List<String> result = new ArrayList<>();
            models.forEach((application, artifacts) -> {
                var appName = application.toString();
                artifacts.forEach(artifact -> {
                    logger.info("{} {}", appName, artifact);
                    result.add(appName + ":" + artifact.getName() + ":" + artifact.getVersion());
                });
            });

            return result;
        } catch (IOException | ModelNotFoundException e) {
            logger.error("Prediction was not go well.", e);
            throw new RuntimeException(e);
        }
    }

    Model makeModel() {
        Block classifier = getBlock();

        Model model = Model.newInstance("RestaurantQuestionsTag");
        model.setBlock(classifier);

        return model;
    }

    private Block getBlock() {
        return new SequentialBlock()
                // text embedding layer
                .add(
                        ndList -> {
                            NDArray data = ndList.singletonOrThrow();
                            NDList inputs = new NDList();
                            long batchSize = data.getShape().get(0);
                            float maxLength = data.getShape().get(1);
                            logger.info("batchSize {} maxLength {}", batchSize, maxLength);
                            if ("PyTorch".equals(Engine.getInstance().getEngineName())) {
                                inputs.add(data.toType(DataType.INT64, false));
                                inputs.add(data.getManager().full(data.getShape(), 1, DataType.INT64));
                                var arrange = data.getManager().arange(maxLength);
                                inputs.add(
                                        arrange.toType(DataType.INT64, false).broadcast(data.getShape())
                                );
                                arrange.close();
                            } else {
                                inputs.add(data);
                                inputs.add(data.getManager().full(new Shape(batchSize), maxLength));
                            }
                            // run embedding
                            try (var embedded = this.loadWordEmbedding()) {
                                return embedded.newPredictor().predict(inputs);
                            } catch (TranslateException e) {
                                throw new IllegalArgumentException("embedding error", e);
                            }
                        })
                // classification layer
                .add(Linear.builder().setUnits(768).build()) // pre classifier 768
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.2f).build())
                .add(Linear.builder().setUnits(5).build()) // 5 category
                .addSingleton(nd -> nd.get(":,0"));

    }

    ZooModel<NDList, NDList> loadWordEmbedding() {
        // MXNet base model
        String modelUrls = "https://resources.djl.ai/test-models/distilbert.zip";
        if ("PyTorch".equals(Engine.getInstance().getEngineName())) {
            modelUrls = "https://resources.djl.ai/test-models/traced_distilbert_wikipedia_uncased.zip";
        }

//        var localPathModel = "./build/model/bert/traced_distilbert_wikipedia_uncased.zip";
//        var path = Path.of(localPathModel);
//        try {
//            if(Files.notExists(path)) {
//                DownloadUtils.download(modelUrls, localPathModel, new ProgressBar());
//            }
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .optApplication(Application.NLP.WORD_EMBEDDING)
                .setTypes(NDList.class, NDList.class)
                .optModelUrls(modelUrls)
//                .optEngine("MXNet")
                //                .optFilter("modelType", "distilbert")
                .optFilter("backbone", "bert")
                .optEngine("PyTorch")
                .optProgress(new ProgressBar())
                .build();
        try {
            return criteria.loadModel();
        } catch (IOException | ModelNotFoundException |
                 MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }
}
