package com.thinksky.classification;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.tabular.CsvDataset;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
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
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import org.apache.commons.csv.CSVFormat;

import javax.enterprise.context.ApplicationScoped;
import java.io.IOException;

@ApplicationScoped
public class PrepareData {

    public PrepareData() {
    }

    public CsvDataset getDataset(int batchSize, BertFullTokenizer tokenizer, int maxLength, int limit) {
        String amazonReview =
                "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Software_v1_00.tsv.gz";
        float paddingToken = tokenizer.getVocabulary().getIndex("[PAD]");

        return CsvDataset.builder()
                .optCsvUrl(amazonReview) // load from Url
                .setCsvFormat(CSVFormat.TDF.withQuote(null).withHeader()) // Setting TSV loading format
                //CSVFormat.Builder.create().setQuote(null).setHeader().build())
                .setSampling(batchSize, true) // make sample size and random access
                .optLimit(limit)
                .addFeature(
                        new Feature(
                                "review_body", new BertFeaturizer(tokenizer, maxLength)))
                .addLabel(
                        new Feature(
                                "star_rating", (buf, data) -> buf.put(Float.parseFloat(data) - 1.0f)))
                .optDataBatchifier(
                        PaddingStackBatchifier.builder()
                                .optIncludeValidLengths(false)
                                .addPad(0, 0, (m) -> m.ones(new Shape(1)).mul(paddingToken))
                                .build()) // define how to pad dataset to a fix length
                .build();
    }

    public Model makeModel() {
        var embedding = loadWordEmbedding();

        Predictor<NDList, NDList> embedder = embedding.newPredictor();

        Block classifier = new SequentialBlock()
                // text embedding layer
                .add(
                        ndList -> {
                            NDArray data = ndList.singletonOrThrow();
                            NDList inputs = new NDList();
                            long batchSize = data.getShape().get(0);
                            float maxLength = data.getShape().get(1);

                            if ("PyTorch".equals(Engine.getInstance().getEngineName())) {
                                inputs.add(data.toType(DataType.INT64, false));
                                inputs.add(data.getManager().full(data.getShape(), 1, DataType.INT64));
                                inputs.add(data.getManager().arange(maxLength)
                                        .toType(DataType.INT64, false)
                                        .broadcast(data.getShape()));
                            } else {
                                inputs.add(data);
                                inputs.add(data.getManager().full(new Shape(batchSize), maxLength));
                            }
                            // run embedding
                            try {
                                return embedder.predict(inputs);
                            } catch (TranslateException e) {
                                throw new IllegalArgumentException("embedding error", e);
                            }
                        })
                // classification layer
                .add(Linear.builder().setUnits(768).build()) // pre classifier
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.2f).build())
                .add(Linear.builder().setUnits(5).build()) // 5 star rating
                .addSingleton(nd -> nd.get(":,0")); // Take [CLS] as the head

        Model model = Model.newInstance("AmazonReviewRatingClassification");
        model.setBlock(classifier);

        return model;

    }

    public ZooModel<NDList, NDList> loadWordEmbedding() {
        // MXNet base model
        String modelUrls = "https://resources.djl.ai/test-models/distilbert.zip";
        if ("PyTorch".equals(Engine.getInstance().getEngineName())) {
            modelUrls = "https://resources.djl.ai/test-models/traced_distilbert_wikipedia_uncased.zip";
        }

        Criteria<NDList, NDList> criteria = Criteria.builder()
                .optApplication(Application.NLP.WORD_EMBEDDING)
                .setTypes(NDList.class, NDList.class)
                .optModelUrls(modelUrls)
                .optFilter("requires_grad", "True")
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
